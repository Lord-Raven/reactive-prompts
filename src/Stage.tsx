import {ReactElement} from "react";
import {StageBase, StageResponse, InitialData, Message} from "@chub-ai/stages-ts";
import {LoadResponse} from "@chub-ai/stages-ts/dist/types/load";
import {Character, User} from "@chub-ai/stages-ts";
import {env, pipeline} from '@xenova/transformers';
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
    config: any;

    constructor(data: InitialData<InitStateType, ChatStateType, MessageStateType, ConfigType>) {
        super(data);
        const {
            characters,
            users,
            config,
        } = data;

        console.log('Config loaded:');
        console.log(config);
        this.config = config;

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

        let yamlResponse = await fetch('chub_meta.yaml');
        const data: any = yaml.load(await yamlResponse.text());

        const inputConceptPrompts: ConceptEntry[] = JSON.parse((this.config ? this.config.inputConcepts : null) ?? data.config_schema.properties.inputConcepts.value);
        const responseConceptPrompts: ConceptEntry[] = JSON.parse((this.config ? this.config.responseConcepts : null) ?? data.config_schema.properties.responseConcepts.value);
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
            pipelineResponse.labels.forEach((value: string, index: number) => this.lastInputWeights[value] = pipelineResponse.scores[index]);
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
        console.log(`Including stage directions: ${stageDirections}`)

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
        console.log(pipelineResponse);
        if (pipelineResponse && pipelineResponse.labels) {
            pipelineResponse.labels.forEach((value: string, index: number) => this.lastResponseWeights[value] = pipelineResponse.scores[index]);
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
