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
  GoogleGenAI,
  GoogleGenAIOptions,
} from '@google/genai';
import { ContentGenerator } from '../core/contentGenerator.js';

export class GeminiProvider implements ContentGenerator {
  private readonly googleGenAI: GoogleGenAI;

  constructor(
    apiKey: string | undefined,
    vertexai: boolean | undefined,
    httpOptions: GoogleGenAIOptions,
  ) {
    this.googleGenAI = new GoogleGenAI({
      apiKey: apiKey === '' ? undefined : apiKey,
      vertexai,
      httpOptions,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    return this.googleGenAI.models.generateContent(request);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    return this.googleGenAI.models.generateContentStream(request);
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    return this.googleGenAI.models.countTokens(request);
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    return this.googleGenAI.models.embedContent(request);
  }
}
