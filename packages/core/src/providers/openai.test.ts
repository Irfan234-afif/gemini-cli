import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenAIProvider } from './openai.js';
import {
  GenerateContentParameters,
  Content,
  Part,
  GenerateContentResponse,
} from '@google/genai';

// Mock openai sdk
const mockCreate = vi.fn();

vi.mock('openai', () => {
  return {
    __esModule: true,
    default: vi.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCreate,
        },
      },
      embeddings: {
        create: vi.fn(),
      },
    })),
  };
});

// Mock logger.server.js used inside provider
vi.mock('./logger.server.js', () => ({
  __esModule: true,
  default: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

describe('OpenAIProvider', () => {
  const apiKey = 'test-key';
  let provider: OpenAIProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    provider = new OpenAIProvider(apiKey);
  });

  it('should forward messages and map functionDeclarations from ToolRegistry', async () => {
    // Build a ToolRegistry and register a simple tool
    const { ToolRegistry } = await import('../tools/tool-registry.js');
    const { BaseTool } = await import('../tools/tools.js');
    const { Type } = await import('@google/genai');

    class DummyTool extends BaseTool<Record<string, unknown>, { llmContent: string; returnDisplay: string }> {
      constructor() {
        super(
          'dummy_tool',
          'dummy_tool',
          'Desc',
          {
            type: Type.OBJECT,
            properties: {
              msg: { type: Type.STRING },
            },
          },
        );
      }
      async execute() {
        return { llmContent: 'ok', returnDisplay: 'ok' };
      }
    }

    const toolRegistry = new ToolRegistry({} as any);
    toolRegistry.registerTool(new DummyTool());

    const params: GenerateContentParameters = {
      contents: [
        { role: 'user', parts: [{ text: 'Hello' }] } as Content,
      ],
      config: {
        tools: [
          { functionDeclarations: toolRegistry.getFunctionDeclarations() },
        ],
      },
    } as unknown as GenerateContentParameters;

    mockCreate.mockResolvedValueOnce({
      choices: [
        {
          message: {
            content: 'Hi',
          },
          finish_reason: 'stop',
        },
      ],
    });

    await provider.generateContent(params);
    const arg = mockCreate.mock.calls[0][0];
    expect(arg.tools?.[0]?.function?.name).toBe('dummy_tool');
    expect(arg).toHaveProperty('tool_choice', 'auto');
  });

  // it('should NOT include tools when no functionDeclarations provided', async () => {
  //   const params: GenerateContentParameters = {
  //     contents: [
  //       { role: 'user', parts: [{ text: 'Ping' }] } as Content,
  //     ],
  //   } as unknown as GenerateContentParameters;

  //   mockCreate.mockResolvedValueOnce({
  //     choices: [
  //       {
  //         message: {
  //           content: 'Pong',
  //         },
  //         finish_reason: 'stop',
  //       },
  //     ],
  //   });

  //   await provider.generateContent(params);

  //   const arg = mockCreate.mock.calls[0][0];
  //   expect(arg.tools).toBeUndefined();
  // });

  it('should parse tool_calls into functionCall Parts', async () => {
    const params: GenerateContentParameters = {
      contents: [
        { role: 'user', parts: [{ text: 'Call tool' }] } as Content,
      ],
      config: {
        tools: [
          {
            functionDeclarations: [
              {
                name: 'calc',
                description: 'calculator',
                parameters: {
                  type: 'object',
                  properties: { a: { type: 'number' } },
                },
              },
            ],
          },
        ],
      },
    } as unknown as GenerateContentParameters;

    mockCreate.mockResolvedValueOnce({
      choices: [
        {
          message: {
            content: null,
            tool_calls: [
              {
                id: 'call1',
                function: {
                  name: 'calc',
                  arguments: JSON.stringify({ a: 1 }),
                },
              },
            ],
          },
          finish_reason: 'stop',
        },
      ],
    });

    const resp = await provider.generateContent(params);
    const parts = resp.candidates![0]!.content!.parts! as Part[];
    const fcPart = parts.find((p) => (p as any).functionCall);
    expect(fcPart).toBeDefined();
    const fc = (fcPart as any).functionCall;
    expect(fc.name).toBe('calc');
    expect(fc.args).toEqual({ a: 1 });
  });

  it('generateContentStream should yield responses with text parts', async () => {
    const { ToolRegistry } = await import('../tools/tool-registry.js');
    const { BaseTool } = await import('../tools/tools.js');
    const { Type } = await import('@google/genai');

    class DummyTool extends BaseTool<Record<string, unknown>, { llmContent: string; returnDisplay: string }> {
      constructor() {
        super(
          'dummy_tool',
          'dummy_tool',
          'Desc',
          {
            type: Type.OBJECT,
            properties: {
              msg: { type: Type.STRING },
            },
          },
        );
      }
      async execute() {
        return { llmContent: 'ok', returnDisplay: 'ok' };
      }
    }

    const toolRegistry = new ToolRegistry({} as any);
    toolRegistry.registerTool(new DummyTool());
    const functionDeclarations = toolRegistry.getFunctionDeclarations();
    const params: GenerateContentParameters = {
      contents: [
        { role: 'user', parts: [{ text: 'Stream' }] } as Content,
      ],
      config: {
        tools: [
          {
            functionDeclarations: functionDeclarations,
          },
        ],
      },
    } as unknown as GenerateContentParameters;

    // Create async iterable that mimics OpenAI stream
    const streamIterable = {
      async *[Symbol.asyncIterator]() {
        yield { choices: [{ delta: { content: 'A' } }] };
        yield { choices: [{ delta: { content: 'B' } }] };
      },
    };

    mockCreate.mockResolvedValueOnce(streamIterable);

    const gen = await provider.generateContentStream(params);
    const collected: string[] = [];
    for await (const chunk of gen) {
      const txt = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
      if (txt) collected.push(txt);
    }
    expect(collected).toEqual(['A', 'B']);
  });

  it('generateContentStream should accumulate tool_calls and emit final functionCalls', async () => {
    const { ToolRegistry } = await import('../tools/tool-registry.js');
    const { BaseTool } = await import('../tools/tools.js');
    const { Type } = await import('@google/genai');

    class DummyTool extends BaseTool<Record<string, unknown>, { llmContent: string; returnDisplay: string }> {
      constructor() {
        super(
          'calc',
          'calc',
          'calculator',
          {
            type: Type.OBJECT,
            properties: { a: { type: Type.NUMBER } },
            required: ['a'],
          },
        );
      }
      async execute() {
        return { llmContent: 'ok', returnDisplay: 'ok' };
      }
    }

    const registry = new ToolRegistry({} as any);
    registry.registerTool(new DummyTool());

    const params: GenerateContentParameters = {
      contents: [
        { role: 'user', parts: [{ text: 'use calc' }] } as Content,
      ],
      config: {
        tools: [{ functionDeclarations: registry.getFunctionDeclarations() }],
      },
    } as unknown as GenerateContentParameters;

    // Create stream with incremental tool_calls
    const streamIterable = {
      async *[Symbol.asyncIterator]() {
        yield {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    id: 'call1',
                    function: { name: 'calc', arguments: '{"a":' },
                  },
                ],
              },
            },
          ],
        };
        yield {
          choices: [
            {
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: { arguments: '1}' },
                  },
                ],
              },
            },
          ],
        };
        // finish chunk
        yield {
          choices: [
            {
              finish_reason: 'tool_calls',
              delta: {},
            },
          ],
        };
      },
    };

    mockCreate.mockResolvedValueOnce(streamIterable);

    const gen = await provider.generateContentStream(params);
    const events: GenerateContentResponse[] = [];
    for await (const chunk of gen) {
      events.push(chunk);
    }

    // Should have at least 1 event with functionCalls
    const finalEvent = events.find((e) => e.functionCalls);
    console.log("finalEvent", finalEvent);
    expect(finalEvent).toBeDefined();
    expect(finalEvent!.functionCalls?.[0].name).toBe('calc');
    expect(finalEvent!.functionCalls?.[0].args).toEqual({ a: 1 });
  });
}); 