# AI Assistant Development Workshop

A hands-on, three-hour beginner workshop on building an AI assistant step by
step. Participants start with a no-code assistant and a good system prompt,
learn how tokens, context windows and temperature affect it, then move to the
API in Python. They add a knowledge base with retrieval-augmented generation
(RAG), check and clean it, give the assistant tools, build an email triage
workflow with a human in the loop, test everything with a golden set, and
publish the assistant as a web app on Hugging Face Spaces. It ends with a
checklist to go through before an assistant goes live.

**No programming experience needed.** All scripts are given. Participants
run them, read them and change small parts. An optional part adds monitoring
with real conversation logs and turns a tool into an MCP server.

---

## Getting Started

**Requirements:** a Google account (for Gemini and a free API key from
[Google AI Studio](https://aistudio.google.com)), Python 3.10+,
[Visual Studio Code](https://code.visualstudio.com), and a free
[Hugging Face](https://huggingface.co) account. GitHub Copilot in VS Code is
only needed for optional Task 12.

Participants **do not need this repository**. They create their own
`assistant-lab` project folder and copy each file from the workshop text into
it, as explained in the workshop's *Before the Workshop* section. The files in
this repository are the finished versions, for instructors and for checking.

Work through the `.md` file: read a section, try the task yourself, then open
the solution to compare.

---

## Workshop

### Building AI Assistants — From System Prompt to Live App
**Workshop:** [AI_ASSISTANT_BASICS.md](AI_ASSISTANT_BASICS.md)

**Reference files** (participants create these themselves during the tasks):

| Task | Files |
|---|---|
| Setup | [config.py](config.py) · [check_setup.py](check_setup.py) |
| 1 | [system_prompt_template.md](system_prompt_template.md) · [system_prompt.md](system_prompt.md) (a possible result) · [knowledge/](knowledge/) · [kurbelwerk-assistant/SKILL.md](kurbelwerk-assistant/SKILL.md) |
| 3 | [hello_api.py](hello_api.py) · [chat.py](chat.py) |
| 5 | [kb.py](kb.py) · [ask.py](ask.py) |
| 6 | [golden_retrieval.json](golden_retrieval.json) · [retrieval_check.py](retrieval_check.py) · [extra/old_price_list.md](extra/old_price_list.md) |
| 7 | [repair_tools.py](repair_tools.py) · [assistant.py](assistant.py) |
| 8 | [inbox.json](inbox.json) · [triage.py](triage.py) |
| 9 | [golden_set.json](golden_set.json) · [evaluate.py](evaluate.py) |
| 10 | [app.py](app.py) · [requirements.txt](requirements.txt) |
| Optional | [log_report.py](log_report.py) (Task 11) · [mcp_server.py](mcp_server.py) (Task 12) |

| Time | Part |
|---|---|
| 0:00 – 0:15 | What an AI assistant is, how an LLM produces text, tokens, context window, system prompt, temperature |
| 0:15 – 0:30 | A first assistant without code: system prompt template and knowledge files |
| 0:30 – 0:40 | Tokens, context window and temperature in the AI Studio playground |
| 0:40 – 1:00 | From the chat window to the API: requests, keys, chat history, streaming |
| 1:00 – 1:10 | Three providers, one assistant: comparing APIs and choosing a provider |
| 1:10 – 1:35 | A knowledge base with RAG: chunks, embeddings, search, citations |
| 1:35 – 1:50 | Retrieval quality: test questions, chunk size, cleaning the knowledge base |
| 1:50 – 2:10 | Tools: function calling, tool errors, what belongs where, MCP |
| 2:10 – 2:25 | Email triage with structured output and a human in the loop |
| 2:25 – 2:40 | Evaluation: a golden set and regression checks |
| 2:40 – 3:00 | Going live on Hugging Face Spaces and the pre-flight checklist |
| Optional | Monitoring with logs and feedback, the repair lookup as an MCP server |

The business in all examples, *Kurbelwerk* in *Musterstadt*, is invented, as
are all emails, ticket numbers and contact details. The Python examples use
the Google Gemini API because it has a free plan; Task 4 shows the same
request for OpenAI and Anthropic.
