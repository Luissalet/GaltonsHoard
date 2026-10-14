"""Harnesses: the ways of talking to a model so that it uses tools. Each one is a small adapter around the same agent loop.

* ``native_tools``: the server's function calling (``tools`` in the request, ``tool_calls`` in the reply, ``tool`` messages back).
* ``json_text``: the model writes a JSON object ``{"tool": ..., "args": {...}}`` in its text.
* ``xml_text``: the model writes ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` (the Qwen / Hermes convention).
* ``react_text``: ``Thought`` / ``Action`` / ``Action Input`` / ``Observation`` lines, ending with ``Final Answer``.

Variants are suffixes of the id: ``.long`` (a long system prompt with rules instead of the short protocol only) and ``.trim`` (observations older
than the last two are replaced by a note before each request). ``json_text.long.trim`` is a valid harness id.

A text without a tool call is the final answer. What the harness cannot parse, or refuses before running anything (an unknown tool, a missing
argument, arguments that are not valid JSON), is a format failure: the model is told and may try again, and the bench counts it apart from a
call that ran and was merely wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..backends import Completion
from ..errors import GaltonError
from .world import ToolSpec

BASES = ("native_tools", "json_text", "xml_text", "react_text")
FLAGS = ("short", "long", "trim")
KEEP_OBSERVATIONS = 2          # the ``.trim`` variants keep this many of the latest observations in full


@dataclass
class Call:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class Rejected:
    """A tool call the harness refuses before running it."""

    reason: str
    name: str = ""
    id: str = ""


@dataclass
class Turn:
    calls: list[Call] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    final: Optional[str] = None            # the final answer, when the reply carried no call
    format_error: str = ""                 # the whole reply looked like a call and could not be read


# ------------------------------------------------------------------------------------------------- parsing helpers
_DECODER = json.JSONDecoder()


def json_objects(text: str) -> list[Any]:
    """Every JSON object found in a text, in order (fences and prose around them are ignored)."""
    out, i = [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            value, end = _DECODER.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        out.append(value)
        i = end


def _args_of(raw: Any) -> tuple[Optional[dict[str, Any]], str]:
    if raw is None:
        return {}, ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return None, "the arguments are not valid JSON"
    if not isinstance(raw, dict):
        return None, "the arguments must be a JSON object"
    return raw, ""


def _tool_list(specs: list[ToolSpec]) -> str:
    lines = []
    for spec in specs:
        params = ", ".join(f"{k}: {v['type']}" + ("" if k in spec.required() else " (optional)") for k, v in spec.parameters["properties"].items())
        lines.append(f"- {spec.name}({params}): {spec.description}")
    return "\n".join(lines)


RULES = ("Rules:\n"
         "1. Call one tool at a time when the next call depends on the result of the previous one.\n"
         "2. Use the exact tool names and argument names listed above. Never invent a tool.\n"
         "3. Do not call a tool when you can answer directly, and do not repeat a call whose result you already have.\n"
         "4. If a tool returns an error, read the message, correct the call and try again.\n"
         "5. When you have what you need, give the final answer in one or two plain sentences with the exact numbers and nothing else.")


# ------------------------------------------------------------------------------------------------- the adapters
@dataclass
class Harness:
    base: str
    prompt: str = "short"
    compact: str = "none"
    native: bool = False

    @property
    def id(self) -> str:
        return self.base + (".long" if self.prompt == "long" else "") + (".trim" if self.compact == "trim" else "")

    # -- what is sent -----------------------------------------------------------------------------------------------
    def system_prompt(self, specs: list[ToolSpec]) -> str:
        raise NotImplementedError

    def tools_payload(self, specs: list[ToolSpec]) -> list[dict[str, Any]]:
        return []

    def reminder(self) -> str:
        return ""

    # -- what comes back --------------------------------------------------------------------------------------------
    def parse(self, comp: Completion) -> Turn:
        raise NotImplementedError

    def assistant_message(self, comp: Completion, turn: Turn, api: str) -> dict[str, Any]:
        return {"role": "assistant", "content": comp.text}

    def observation(self, results: list[tuple[Any, str]], api: str, *, omit: bool = False) -> list[dict[str, Any]]:
        """Messages that give the results back. ``results`` are ``(Call or Rejected, text)``; ``omit`` writes the older-observation note instead."""
        raise NotImplementedError

    def format_error_message(self, reason: str) -> dict[str, Any]:
        return {"role": "user", "content": f"Format error: {reason}. {self.reminder()}".strip()}

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "base": self.base, "prompt": self.prompt, "compact": self.compact, "native": self.native, "external": False}


def _omitted(text: str) -> str:
    return f"[older result omitted: {len(text)} characters]"


class NativeTools(Harness):
    def __init__(self, prompt: str = "short", compact: str = "none"):
        super().__init__("native_tools", prompt, compact, native=True)

    def system_prompt(self, specs: list[ToolSpec]) -> str:
        base = "You are an assistant that can call tools. Use a tool when you need information or an action; otherwise answer directly. When you have the answer, reply with the final answer only."
        return base + ("\n\n" + RULES.replace("listed above", "provided") if self.prompt == "long" else "")

    def tools_payload(self, specs: list[ToolSpec]) -> list[dict[str, Any]]:
        return [s.schema() for s in specs]

    def reminder(self) -> str:
        return "Call the tool through the function-calling interface, or answer in plain text."

    def parse(self, comp: Completion) -> Turn:
        turn = Turn()
        for i, call in enumerate(comp.tool_calls):
            args = call.get("arguments", {})
            if isinstance(args, dict) and "_unparsed" in args:
                turn.rejected.append(Rejected("the arguments are not valid JSON", call["name"], f"call_{i}"))
            else:
                turn.calls.append(Call(call["name"], args if isinstance(args, dict) else {}, f"call_{i}"))
        if turn.calls or turn.rejected:
            return turn
        text = comp.text.strip()
        if "<tool_call>" in text or re.match(r'^\{\s*"name"\s*:', text):
            turn.format_error = "the tool call was written in the text, not through the function-calling interface"
        else:
            turn.final = text
        return turn

    def assistant_message(self, comp: Completion, turn: Turn, api: str) -> dict[str, Any]:
        calls = []
        for item in [*turn.calls, *turn.rejected]:
            name = item.name
            args = item.args if isinstance(item, Call) else {}
            fn: dict[str, Any] = {"name": name, "arguments": args if api == "ollama" else json.dumps(args, ensure_ascii=False)}
            calls.append({"id": item.id, "type": "function", "function": fn})
        msg: dict[str, Any] = {"role": "assistant", "content": comp.text}
        if calls:
            msg["tool_calls"] = calls
        return msg

    def observation(self, results: list[tuple[Any, str]], api: str, *, omit: bool = False) -> list[dict[str, Any]]:
        out = []
        for item, text in results:
            msg = {"role": "tool", "tool_call_id": item.id, "content": _omitted(text) if omit else text}
            if api == "ollama":
                msg["tool_name"] = item.name
            out.append(msg)
        return out


class JsonText(Harness):
    def __init__(self, prompt: str = "short", compact: str = "none"):
        super().__init__("json_text", prompt, compact)

    def system_prompt(self, specs: list[ToolSpec]) -> str:
        text = ("You are an assistant that can use tools.\n\nTools:\n" + _tool_list(specs) + "\n\n"
                'To call a tool, reply with only a JSON object: {"tool": "<name>", "args": {<arguments>}}. You will get the result in the next message and may call another tool. '
                "When you have the answer, reply in plain text without any JSON.")
        return text + ("\n\n" + RULES if self.prompt == "long" else "")

    def reminder(self) -> str:
        return 'Reply with a JSON object {"tool": "<name>", "args": {...}} to call a tool, or with plain text for the final answer.'

    def parse(self, comp: Completion) -> Turn:
        text = comp.text.strip()
        turn = Turn()
        for obj in json_objects(text):
            if not isinstance(obj, dict):
                continue
            if "tool" in obj:
                name = obj["tool"]
                args, why = _args_of(obj.get("args", {}))
                if not isinstance(name, str) or not name.strip():
                    turn.rejected.append(Rejected("the 'tool' value must be the name of a tool"))
                elif args is None:
                    turn.rejected.append(Rejected(why, name))
                else:
                    turn.calls.append(Call(name.strip(), args))
            elif {"name", "function", "arguments", "args", "tool_name", "tool_calls"} & set(obj):
                turn.rejected.append(Rejected("the JSON object has no 'tool' key"))
        if turn.calls or turn.rejected:
            return turn
        if re.search(r'"tool"\s*:', text):
            turn.format_error = "the JSON could not be parsed"
        else:
            turn.final = text
        return turn

    def observation(self, results: list[tuple[Any, str]], api: str, *, omit: bool = False) -> list[dict[str, Any]]:
        body = "\n\n".join(f"Tool result ({item.name or 'call'}):\n" + (_omitted(text) if omit else text) for item, text in results)
        return [{"role": "user", "content": body}]


class XmlText(Harness):
    PATTERN = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)

    def __init__(self, prompt: str = "short", compact: str = "none"):
        super().__init__("xml_text", prompt, compact)

    def system_prompt(self, specs: list[ToolSpec]) -> str:
        signatures = "\n".join(json.dumps(s.schema(), ensure_ascii=False) for s in specs)
        text = ("You are a helpful assistant. You may call one or more functions to assist with the user query.\n\n"
                f"The function signatures are inside <tools></tools>:\n<tools>\n{signatures}\n</tools>\n\n"
                "For each function call, return a JSON object with the function name and arguments inside <tool_call></tool_call> tags:\n"
                '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'
                "The result comes back inside <tool_response></tool_response> tags. When you have the answer, reply in plain text without tags.")
        return text + ("\n\n" + RULES if self.prompt == "long" else "")

    def reminder(self) -> str:
        return 'Write <tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call> to call a tool, or plain text for the final answer.'

    def parse(self, comp: Completion) -> Turn:
        text = comp.text.strip()
        turn = Turn()
        if "<tool_call>" not in text:
            turn.final = text
            return turn
        for payload in self.PATTERN.findall(text):
            objs = [o for o in json_objects(payload) if isinstance(o, dict)]
            if not objs:
                turn.rejected.append(Rejected("the content of <tool_call> is not valid JSON"))
                continue
            obj = objs[0]
            name = obj.get("name")
            args, why = _args_of(obj.get("arguments", obj.get("args")))
            if not isinstance(name, str) or not name.strip():
                turn.rejected.append(Rejected("the JSON inside <tool_call> has no 'name'"))
            elif args is None:
                turn.rejected.append(Rejected(why, name))
            else:
                turn.calls.append(Call(name.strip(), args))
        return turn

    def observation(self, results: list[tuple[Any, str]], api: str, *, omit: bool = False) -> list[dict[str, Any]]:
        body = "\n".join(f"<tool_response>\n{_omitted(text) if omit else text}\n</tool_response>" for _, text in results)
        return [{"role": "user", "content": body}]


class ReactText(Harness):
    def __init__(self, prompt: str = "short", compact: str = "none"):
        super().__init__("react_text", prompt, compact)

    def system_prompt(self, specs: list[ToolSpec]) -> str:
        names = ", ".join(s.name for s in specs)
        text = ("Answer the question as best you can. You have access to these tools:\n\n" + _tool_list(specs) + "\n\n"
                "Use this format:\n\nThought: what to do next\nAction: one of [" + names + "]\nAction Input: a JSON object with the arguments\n"
                "Observation: the result (it is given to you; never write it yourself)\n... (repeat Thought/Action/Action Input/Observation as needed)\n"
                "Thought: I now know the final answer\nFinal Answer: the answer")
        return text + ("\n\n" + RULES if self.prompt == "long" else "")

    def reminder(self) -> str:
        return "Use 'Action:' and 'Action Input:' (a JSON object) to call a tool, or 'Final Answer:' to finish."

    def parse(self, comp: Completion) -> Turn:
        text = re.split(r"(?im)^\s*Observation\s*:", comp.text)[0].strip()      # whatever follows a made-up observation is not read
        turn = Turn()
        final = re.search(r"Final Answer\s*:\s*(.*)", text, re.S | re.I)
        action = re.search(r"^\s*Action\s*:\s*(.+)$", text, re.M | re.I)
        if action and final:
            turn.format_error = "the reply has both an Action and a Final Answer"
            return turn
        if not action:
            turn.final = final.group(1).strip() if final else text
            return turn
        name = action.group(1).strip().strip("`'\"[]* ")
        given = re.search(r"Action Input\s*:\s*(.*)", text, re.S | re.I)
        if given is None:
            turn.rejected.append(Rejected("there is no 'Action Input:' line", name))
            return turn
        objs = [o for o in json_objects(given.group(1)) if isinstance(o, dict)]
        if not objs:
            turn.rejected.append(Rejected("the Action Input is not a JSON object", name))
        else:
            turn.calls.append(Call(name, objs[0]))
        return turn

    def assistant_message(self, comp: Completion, turn: Turn, api: str) -> dict[str, Any]:
        return {"role": "assistant", "content": re.split(r"(?im)^\s*Observation\s*:", comp.text)[0].strip()}

    def observation(self, results: list[tuple[Any, str]], api: str, *, omit: bool = False) -> list[dict[str, Any]]:
        return [{"role": "user", "content": "\n".join("Observation: " + (_omitted(text) if omit else text) for _, text in results)}]


_CLASSES = {"native_tools": NativeTools, "json_text": JsonText, "xml_text": XmlText, "react_text": ReactText}
DEFAULT_HARNESSES = list(BASES)


def parse_harness_id(ref: str) -> Harness:
    """``json_text``, ``xml_text.long``, ``native_tools.long.trim``... The variants may come in any order; the id of the result is canonical."""
    tokens = [t for t in str(ref).strip().lower().replace("+", ".").split(".") if t]
    if not tokens or tokens[0] not in _CLASSES:
        raise GaltonError("invalid", "hb_unknown_harness", ref=ref, options=", ".join(BASES))
    flags = tokens[1:]
    bad = [f for f in flags if f not in FLAGS]
    if bad or ("short" in flags and "long" in flags):
        raise GaltonError("invalid", "hb_unknown_harness", ref=ref, options=", ".join(BASES) + " + .long .trim")
    return _CLASSES[tokens[0]]("long" if "long" in flags else "short", "trim" if "trim" in flags else "none")


def trim_history(entries: list[dict[str, Any]], keep: int = KEEP_OBSERVATIONS) -> list[dict[str, Any]]:
    """The messages to send: when ``keep`` is set, every observation except the last ``keep`` ones is replaced by its short note."""
    obs = [i for i, e in enumerate(entries) if e["kind"] == "obs"]
    old = set(obs[: max(0, len(obs) - keep)]) if keep else set()
    return [(e["short"] if i in old else e["msg"]) if e["kind"] == "obs" else e["msg"] for i, e in enumerate(entries)]
