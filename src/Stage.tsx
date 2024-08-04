import {ReactElement} from "react";
import {StageBase, StageResponse, InitialData, Message} from "@chub-ai/stages-ts";
import {LoadResponse} from "@chub-ai/stages-ts/dist/types/load";
import {Character, User} from "@chub-ai/stages-ts";
import {env, pipeline} from '@xenova/transformers';
import * as fs from 'fs';
import * as yaml from 'js-yaml';

type MessageStateType = any;

type ConfigType = any;

type InitStateType = any;

type ChatStateType = any;

type ConceptEntry = {
    concept: string,
    threshold: number,
    prompt: string
}


export class Stage extends StageBase<InitStateType, ChatStateType, MessageStateType, ConfigType> {

    readonly INPUT_CONCEPTS: string = `[{` +
        `"concept":"focused",` +
        `"threshold":0.8,` +
        `"prompt":"Invent or incorporate relevant or flavorful details surrounding the object of {{user}}'s attention."` +
        `},{` +
        `"concept":"narrow narrative potential",` +
        `"threshold":0.6,` +
        `"prompt":"This is a tight moment; write only two or three sentences in your response."` +
        `},{` +
        `"concept":"intense action",` +
        `"threshold":0.7,` +
        `"prompt":"Directly address and describe the outcome or consequences of {{user}}'s actions."` +
        `},{` +
        `"concept":"high narrative potential",` +
        `"threshold":0.6,` +
        `"prompt":"This is an open-ended moment; write about two paragraphs in your response."` +
        `},{` +
        `"concept":"engaged",` +
        `"threshold":0.8,` +
        `"prompt":"{{user}} is engaged in the current scene; keep this moment going."` +
        `},{` +
        `"concept":"disengaged",` +
        `"threshold":0.6,` +
        `"prompt":"{{user}} is disengaging from the current scene; move events forward."` +
        `}]`;

    readonly RESPONSE_CONCEPTS: string = `[{\n` +
        `        "concept":"flowery",\n` +
        `        "threshold": 0.8,\n` +
        `        "prompt":"Keep your prose more grounded and concise."\n` +
        `        }]`;

    // messageState
    lastInputWeights: {[key: string]: number};
    lastResponseWeights: {[key: string]: number};

    // other
    characters: {[key: string]: Character};
    user: User;
    conceptPipeline: any;
    inputConcepts: string[] = [];
    inputThresholds: {[key: string]: number} = {};
    inputPrompts: {[key: string]: string} = {};
    responseConcepts: string[] = [];
    responseThresholds: {[key: string]: number} = {};
    responsePrompts: {[key: string]: string} = {};

    constructor(data: InitialData<InitStateType, ChatStateType, MessageStateType, ConfigType>) {
        super(data);
        const {
            characters,
            users,
            config,
        } = data;

        console.log('Config loaded:');
        console.log(config);

        const fileContents = fs.readFileSync('chub_meta.yaml', 'utf8');
        const yamlData = yaml.load(fileContents) as Record<string, any>;
        console.log(yamlData);

        const inputConceptPrompts: ConceptEntry[] = JSON.parse((config ? config.inputConcepts : null) ?? this.INPUT_CONCEPTS);
        const responseConceptPrompts: ConceptEntry[] = JSON.parse((config ? config.responseConcepts : null) ?? this.RESPONSE_CONCEPTS);
        for (let entry of inputConceptPrompts) {
            this.inputConcepts.push(entry.concept);
            this.inputThresholds[entry.concept] = entry.threshold;
            this.inputPrompts[entry.concept] = entry.prompt;
        }
        for (let entry of responseConceptPrompts) {
            this.responseConcepts.push(entry.concept);
            this.responseThresholds[entry.concept] = entry.threshold;
            this.responsePrompts[entry.concept] = entry.prompt;
        }

        this.lastInputWeights = {};
        this.lastResponseWeights = {};

        this.characters = characters;
        this.user = users[Object.keys(users)[0]];

        this.conceptPipeline = null;
        env.allowRemoteModels = false;
    }

    async load(): Promise<Partial<LoadResponse<InitStateType, ChatStateType, MessageStateType>>> {

        try {
            this.conceptPipeline = await pipeline("zero-shot-classification", "Xenova/mobilebert-uncased-mnli");//"SamLowe/roberta-base-go_emotions");
        } catch (exception: any) {
            console.error(`Error loading pipeline: ${exception}`);
        }

        return {
            success: true,
            error: null,
            initState: null,
            chatState: null,
            messageState: this.writeMessageState()
        };
    }

    readMessageState(messageState: MessageStateType) {
        if (messageState) {
            this.lastInputWeights = messageState.lastInputWeights ?? '';
            this.lastResponseWeights = messageState.lastResponseWeights ?? '';
        }
    }

    writeMessageState(): MessageStateType {
        return {
            lastInputWeights: this.lastInputWeights ?? {},
            lastResponseWeights: this.lastResponseWeights ?? {}
        };
    }

    replaceTags(source: string, replacements: {[name: string]: string}) {
        return source.replace(/{{([A-z]*)}}/g, (match) => {
            return replacements[match.substring(2, match.length - 2)];
        });
    }

    async setState(state: MessageStateType): Promise<void> {
        this.readMessageState(state);
    }

    async beforePrompt(userMessage: Message): Promise<Partial<StageResponse<ChatStateType, MessageStateType>>> {
        const {
            content,
            promptForId
        } = userMessage;

        this.lastInputWeights = {}
        let pipelineResponse = await this.conceptPipeline(content, this.inputConcepts, { multi_label: true });
        console.log(pipelineResponse);
        if (pipelineResponse && pipelineResponse.labels) {
            pipelineResponse.labels.forEach((value: string, index: number) => {console.log(value);this.lastInputWeights[value] = pipelineResponse.scores[index];});
        }

        let inputAdditions = Object.entries(this.lastInputWeights)
            .filter(([concept, strength]) => strength >= this.inputThresholds[concept])
            .map(([concept]) => this.inputPrompts[concept])
            .join('\n');

        let responseAdditions = Object.entries(this.lastResponseWeights)
            .filter(([concept, strength]) => strength >= this.responseThresholds[concept])
            .map(([concept]) => this.responsePrompts[concept])
            .join('\n');

        if (inputAdditions) {
            inputAdditions = this.replaceTags(inputAdditions, {
                "user": this.user.name,
                "char": promptForId ? this.characters[promptForId].name : ''
            });
        } else {
            inputAdditions = '';
        }
        if (responseAdditions) {
            responseAdditions = this.replaceTags(responseAdditions, {
                "user": this.user.name,
                "char": promptForId ? this.characters[promptForId].name : ''
            });
        } else {
            responseAdditions = '';
        }

        let stageDirections = `${inputAdditions}\n${responseAdditions}`;

        return {
            stageDirections: stageDirections.trim().length > 0 ? `[INST]${stageDirections}[/INST]` : null,
            messageState: this.writeMessageState(),
            modifiedMessage: null,
            systemMessage: null,
            error: null,
            chatState: null,
        };
    }

    async afterResponse(botMessage: Message): Promise<Partial<StageResponse<ChatStateType, MessageStateType>>> {
        const {
            content
        } = botMessage;

        let pipelineResponse = await this.conceptPipeline(content, this.responsePrompts, { multi_label: true });
        if (pipelineResponse && pipelineResponse.labels) {
            pipelineResponse.labels.forEach((value: string, index: number) => {console.log(value);this.lastResponseWeights[value] = pipelineResponse.scores[index];});
        }

        return {
            stageDirections: null,
            messageState: this.writeMessageState(),
            modifiedMessage: null,
            error: null,
            systemMessage: null,
            chatState: null
        };
    }


    render(): ReactElement {
        return <div style={{
            width: '100vw',
            height: '100vh',
            display: 'grid',
            alignItems: 'stretch'
        }}>

        </div>;
    }

}
