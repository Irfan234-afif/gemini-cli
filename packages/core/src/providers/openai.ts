/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
} from '@google/genai';
import { ContentGenerator } from '../core/contentGenerator.js';
import OpenAI from 'openai';
import { logTools } from './toolLogger.js';
import logger from './logger.server.js';
// The official OpenAI SDK exports detailed types, but the specific path may change between versions.
// To avoid brittle imports, define a minimal local type alias sufficient for our use case.
type ChatCompletionMessageParam = {
  role: 'user' | 'assistant' | 'system' | "tool";
  tool_calls?: {
    id: string;
    type: 'function';
    function: {
      name: string;
      arguments: string;
    };
  }[];
  tool_call_id?: string | undefined,
  content?: string | undefined;
};

// Note: We will dynamically import `@dqbd/tiktoken` only when needed inside `countTokens` to avoid compile-time type issues.

function geminiContentToOpenAI(
  contents: Content[],
): ChatCompletionMessageParam[] {
  const messages: ChatCompletionMessageParam[] = [];
  
  for (const [index, content] of contents.entries()) {
    if (index === 0) {
      messages.push({
        role: "system",
        content: content.parts?.at(0)?.text ?? '',
      });
      continue;
    } else if (index === 1) {
      // Continue to the next content because openai use only system role for the first message
      continue;
    }
    // OpenAI doesn't have a direct equivalent of a "model" role.
    // We'll map "user" to "user" and "model" to "assistant".
    const role = content.role === 'user' ? 'user' : 'assistant';
    if (!content.parts) {
      continue;
    }
    for (const part of content.parts) {
      if ('functionCall' in part && part.functionCall) {
        messages.push({
          role: "assistant",
          tool_calls: [{
            id: part.functionCall.id ?? '',
            type: 'function',
            function: {
              name: part.functionCall.name ?? '',
              arguments: JSON.stringify(part.functionCall.args),
            },
          }],
        });
      }
      if ('functionResponse' in part && part.functionResponse) {
        let content: string = JSON.stringify(part.functionResponse.response);

        messages.push({
          role: "tool",
          content: content ?? '',
          tool_call_id: part.functionResponse.id,
        });
      }
      if ('text' in part) {
        messages.push({
          role,
          content: part.text,
        });
      }
      continue;
    }
  }
  return messages;
}

export class OpenAIProvider implements ContentGenerator {
  private readonly openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey: apiKey, baseURL: "https://openrouter.ai/api/v1" });
  }

  // Helper to convert Gemini FunctionDeclarations to OpenAI function schema
  private _mapFunctionDeclarations(request: GenerateContentParameters): any[] | undefined {
    // Extract function declarations from tools array if present
    // OpenAI expects: { name, description?, parameters }
    // The parameters schema is already JSON Schema compatible, so we can pass as-is
    const tools = (request as any)?.config?.tools as
      | { functionDeclarations?: any[] }[]
      | undefined;
    if (!tools || tools.length === 0) {
      return undefined;
    }

    const functionDeclarations: any[] = [];
    for (const t of tools) {
      if (t && 'functionDeclarations' in t && Array.isArray((t as any).functionDeclarations)) {
        functionDeclarations.push(...(t as any).functionDeclarations);
      }
    }
    if (functionDeclarations.length === 0) {
      return undefined;
    }

    // Helper to deep-convert @google/genai Type enums (STRING, OBJECT, etc.)
    const normalizeSchema = (schema: any): any => {
      if (!schema || typeof schema !== 'object') return schema;
      const newSchema: any = Array.isArray(schema) ? [] : {};
      for (const [k, v] of Object.entries(schema)) {
        if (k === 'type' && typeof v === 'string') {
          newSchema[k] = v.toLowerCase();
        } else if (['minLength','maxLength','minItems','maxItems','minProperties','maxProperties'].includes(k) && typeof v === 'string' && /^\d+$/.test(v)) {
          newSchema[k] = parseInt(v as string, 10);
        } else if (k === 'items') {
          newSchema[k] = normalizeSchema(v);
        } else if (k === 'properties') {
          const props: any = {};
          for (const [pk, pv] of Object.entries(v as Record<string, unknown>)) {
            props[pk] = normalizeSchema(pv);
          }
          newSchema[k] = props;
        } else if (k === 'anyOf' && Array.isArray(v)) {
          newSchema[k] = v.map(normalizeSchema);
        } else {
          newSchema[k] = normalizeSchema(v);
        }
      }
      return newSchema;
    };

    // Convert to OpenAI "tools" array format (function type)
    return functionDeclarations.map((fd) => {
      return {
        type: "function",
        function: {
          name: fd.name,
          description: fd.description,
          parameters: normalizeSchema(fd.parameters ?? { type: "object", properties: {} }),
        },
      };
    });
  }

  private _buildPartsFromOpenAIResponse(choice: any): { parts: Part[]; functionCalls: any[] } {
    const parts: Part[] = [];
    const functionCallsArr: any[] = [];
    const message = choice?.message;
    if (!message) {
      return { parts, functionCalls: functionCallsArr };
    }

    // Text content
    if (message.content) {
      parts.push({ text: message.content });
    }

    // Tool calls (function calls)
    const toolCalls = message.tool_calls || [];
    for (const tc of toolCalls) {
      try {
        const argsObj = tc.function?.arguments
          ? JSON.parse(tc.function.arguments)
          : {};
        parts.push({
          functionCall: {
            id: tc.id,
            name: tc.function?.name,
            args: argsObj,
          } as any, // cast to align with @google/genai FunctionCall type
        });
        functionCallsArr.push({ id: tc.id, name: tc.function?.name, args: argsObj });
      } catch {
        // If parsing fails, skip this tool call
      }
    }

    return { parts, functionCalls: functionCallsArr };
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const messages = geminiContentToOpenAI(
      Array.isArray(request.contents)
        ? (request.contents as Content[])
        : [(request.contents as unknown as Content)],
    );

    const toolsForOpenAI = this._mapFunctionDeclarations(request);
    // Log the tools (if any) to a dedicated log file before the request is sent
    // await logTools(toolsForOpenAI);

    const completion = await this.openai.chat.completions.create({
      messages,
      model: "deepseek/deepseek-chat-v3-0324",
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      top_p: request.config?.topP,
      // Only include tools param if available
      ...(toolsForOpenAI ? { tools: toolsForOpenAI, tool_choice: "auto" } : {}),
    } as any);

    const choice = completion.choices[0];
    const { parts, functionCalls } = this._buildPartsFromOpenAIResponse(choice);

    return {
      candidates: [
        {
          content: {
            parts: parts.length > 0 ? parts : [{ text: "" }],
            role: "model",
          },
          finishReason: choice.finish_reason || "STOP",
          index: 0,
          safetyRatings: [],
        },
      ],
      functionCalls: functionCalls.length ? functionCalls : undefined,
      promptFeedback: {
        safetyRatings: [],
      },
    } as unknown as GenerateContentResponse;
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    logger.info("contents", request.contents, 'contents.log');
    const messages = geminiContentToOpenAI(
      Array.isArray(request.contents)
        ? (request.contents as Content[])
        : [(request.contents as unknown as Content)],
    );

    logger.info("messages", messages, 'messages.log');

    const toolsForOpenAI = this._mapFunctionDeclarations(request);
    // Log the tools (if any) to a dedicated log file before the streaming request is sent
    await logTools(toolsForOpenAI);

    const stream: any = await this.openai.chat.completions.create({
      messages,
      model: "deepseek/deepseek-chat-v3-0324",
      temperature: request.config?.temperature,
      max_tokens: request.config?.maxOutputTokens,
      top_p: request.config?.topP,
      stream: true,
      ...(toolsForOpenAI ? { tools: toolsForOpenAI, tool_choice: "auto" } : {}),
    } as any);

    async function* mapStream(): AsyncGenerator<GenerateContentResponse> {
      // builders keyed by tool_call index (order) or id
      const builders: Record<string, { id: string | undefined; name: string | undefined; args: string } > = {};

      for await (const chunk of stream) {
        const choice = chunk.choices?.[0];

        // Handle delta tool_calls accumulation
        if (choice?.delta?.tool_calls) {
          for (const tc of choice.delta.tool_calls as any[]) {
            const key = (tc.index !== undefined ? String(tc.index) : (tc.id ?? '0'));
            if (!builders[key]) {
              builders[key] = { id: tc.id, name: undefined, args: '' };
            }
            if (tc.function?.name) {
              builders[key].name = tc.function.name;
            }
            if (tc.function?.arguments) {
              builders[key].args += tc.function.arguments;
            }
          }
        }

        // When finish_reason === 'tool_calls' send final functionCalls
        if (choice?.finish_reason === 'tool_calls') {
          const functionCalls = Object.values(builders).map((b) => ({
            id: b.id,
            name: b.name,
            args: b.args ? JSON.parse(b.args) : {},
          }));
          logger.info("functionCalls", functionCalls, 'functionCalls.log');

          yield {
            candidates: [
              {
                content: { parts: [], role: 'model' },
                finishReason: 'STOP',
                index: 0,
                safetyRatings: [],
              },
            ],
            functionCalls,
            promptFeedback: { safetyRatings: [] },
          } as unknown as GenerateContentResponse;

          // reset builders after emitting
          for (const k of Object.keys(builders)) delete builders[k];
          continue;
        }

        // Otherwise, if text delta exists emit text
        const responseText = choice?.delta?.content;
        if (responseText) {
          yield {
            candidates: [
              {
                content: {
                  parts: [{ text: responseText }],
                  role: 'model',
                },
                finishReason: 'STOP',
                index: 0,
                safetyRatings: [],
              },
            ],
            promptFeedback: { safetyRatings: [] },
          } as unknown as GenerateContentResponse;
        }
      }
    }

    return mapStream();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Note: This is a local estimation using tiktoken.
    // The actual token count from the OpenAI API may differ slightly.
    let tokenCount = 0;
    let encoding: { encode: (t: string) => number[]; free: () => void } | null = null;
    try {
      // @ts-ignore - optional dependency typings may be missing
      const tiktokenMod = await import('@dqbd/tiktoken');
      encoding = (tiktokenMod as any).get_encoding
        ? (tiktokenMod as any).get_encoding('cl100k_base')
        : null;
    } catch {
      // Fallback: approximate tokens by dividing character length by 4
    }

    let text = '';
    if (typeof request.contents === 'string') {
      text = request.contents;
    } else {
      const contentsArr = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      text = contentsArr
        .map((item: any) => {
          if (typeof item === 'string') return item;
          if (item?.parts) {
            return item.parts
              .map((p: any) => ('text' in p ? p.text : ''))
              .join('');
          }
          return '';
        })
        .join('\n');
    }

    if (encoding) {
      const tokens = encoding.encode(text);
      encoding.free();
      tokenCount = tokens.length;
    } else {
      tokenCount = Math.ceil(text.length / 4);
    }

    return {
      totalTokens: tokenCount,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const { contents, model } = request;

    const toPlainText = (item: unknown): string => {
      if (typeof item === 'string') return item;
      if (Array.isArray(item)) {
        return (item as Part[])
          .map((p) => ('text' in p ? (p as Part).text : ''))
          .join('');
      }
      if (item && typeof item === 'object' && 'parts' in (item as Content)) {
        return ((item as Content).parts || [])
          .map((p) => ('text' in p ? p.text : ''))
          .join('');
      }
      return '';
    };

    const inputArray: string[] = Array.isArray(contents)
      ? contents.map((c) => toPlainText(c))
      : [toPlainText(contents)];

    const embedding = await this.openai.embeddings.create({
      model: model as string,
      input: inputArray.length === 1 ? inputArray[0] : inputArray,
      encoding_format: 'float',
    });

    return {
      embeddings: embedding.data.map((d) => ({ values: d.embedding })),
    } as unknown as EmbedContentResponse;
  }
}
