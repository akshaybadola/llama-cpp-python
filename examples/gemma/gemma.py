from typing import cast, Optional, Union, Iterator, Literal, Any
import re
import os
import sys
import ctypes
import json
from contextlib import ExitStack
import base64
import urllib.request
from io import BytesIO

from PIL import Image
from jinja2.sandbox import ImmutableSandboxedEnvironment

import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
import llama_cpp.llama_grammar as llama_grammar
import llama_cpp.llava_cpp as llava_cpp
from llama_cpp._utils import suppress_stdout_stderr, Singleton
from llama_cpp.llama_chat_format import (
    _get_system_message,
    _grammar_for_json,
    _grammar_for_json_schema,
    _grammar_for_response_format,
    _convert_completion_to_chat_function,
    _convert_completion_to_chat
)


class Gemma3ChatHandler:
    DEFAULT_SYSTEM_MESSAGE: Optional[str] = (
        "You are a helpful assistant."
    )

    CHAT_FORMAT = (
        "{{ bos_token }}"
        "{%- if messages[0]['role'] == 'system' -%}"
        "{%- if messages[0]['content'] is string -%}"
        "{%- set first_user_prefix = messages[0]['content'] + '\\n' -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = messages[0]['content'][0]['text'] + '\\n' -%}"
        "{%- endif -%}"
        "{%- set loop_messages = messages[1:] -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = \"\" -%}"
        "{%- set loop_messages = messages -%}"
        "{%- endif -%}"
        "{%- for message in loop_messages -%}"
        "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{%- endif -%}"
        "{%- if (message['role'] == 'assistant') -%}"
        "{%- set role = \"model\" -%}"
        "{%- else -%}"
        "{%- set role = message['role'] -%}"
        "{%- endif -%}"
        "{{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else \"\") }}"
        "{%- if message['content'] is string -%}"
        "{{ message['content'] | trim }}"
        "{%- elif message['content'] is iterable -%}"
        "{%- for item in message['content'] -%}"
        "{%- if item['type'] == 'image' -%}"
        "{{ '<start_of_image>' }}"
        "{%- elif item['type'] == 'text' -%}"
        "{{ item['text'] | trim }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- else -%}"
        "{{ raise_exception(\"Invalid content type\") }}"
        "{%- endif -%}"
        "{{ '<end_of_turn>\\n' }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{'<start_of_turn>model\\n'}}"
        "{%- endif -%}"
    )

    def __init__(self, clip_model_path: str, verbose: bool = True):

        self.clip_model_path = clip_model_path
        self.verbose = verbose

        self._llava_cpp = llava_cpp  # TODO: Fix
        self._exit_stack = ExitStack()
        self._last_image_embed: Optional[
            llava_cpp.CtypesPointer[llava_cpp.llava_image_embed]
        ] = None
        self._last_image_hash: Optional[int] = None

        if not os.path.exists(clip_model_path):
            raise ValueError(f"Clip model path does not exist: {clip_model_path}")

        with suppress_stdout_stderr(disable=self.verbose):
            clip_ctx = self._llava_cpp.clip_model_load(self.clip_model_path.encode(), 0)

            if clip_ctx is None:
                raise ValueError(f"Failed to load clip model: {clip_model_path}")

            self.clip_ctx = clip_ctx

            def clip_free():
                with suppress_stdout_stderr(disable=self.verbose):
                    self._llava_cpp.clip_free(self.clip_ctx)

            self._exit_stack.callback(clip_free)

        def last_image_embed_free():
            with suppress_stdout_stderr(disable=self.verbose):
                if self._last_image_embed is not None:
                    self._llava_cpp.llava_image_embed_free(self._last_image_embed)
                    self._last_image_embed = None

        self._exit_stack.callback(last_image_embed_free)

    def load_image(self, image_url: str) -> bytes:
        return self._load_image(image_url)

    def _embed_image_bytes(self, image_bytes: bytes, n_threads_batch: int = 1):
        if (
            self._last_image_embed is not None
            and self._last_image_hash is not None
            and hash(image_bytes) == self._last_image_hash
        ):
            return self._last_image_embed
        with suppress_stdout_stderr(disable=self.verbose):
            # Free the previous image embed
            if self._last_image_embed is not None:
                self._llava_cpp.llava_image_embed_free(self._last_image_embed)
                self._last_image_embed = None
                self._last_image_hash = None
            embed = self._llava_cpp.llava_image_embed_make_with_bytes(
                self.clip_ctx,
                n_threads_batch,
                (ctypes.c_uint8 * len(image_bytes)).from_buffer(
                    bytearray(image_bytes)
                ),
                len(image_bytes),
            )
            self._last_image_embed = embed
            self._last_image_hash = hash(image_bytes)
            return embed

    def __call__(
        self,
        *,
        llama: llama.Llama,
        messages: list[llama_types.ChatCompletionRequestMessage],
        functions: Optional[list[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[list[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, list[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[
            llama_types.ChatCompletionRequestResponseFormat
        ] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logit_bias: Optional[dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:
        assert self.clip_ctx is not None

        system_prompt = _get_system_message(messages)
        if system_prompt == "" and self.DEFAULT_SYSTEM_MESSAGE is not None:
            messages = [
                llama_types.ChatCompletionRequestSystemMessage(
                    role="system", content=self.DEFAULT_SYSTEM_MESSAGE
                )
            ] + messages

        image_urls = self.get_image_urls(messages)
        template = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(self.CHAT_FORMAT)
        text = template.render(
            messages=messages,
            add_generation_prompt=True,
            eos_token=llama.detokenize([llama.token_eos()]),
            bos_token=llama.detokenize([llama.token_bos()]),
        )
        split_text = self.split_text_on_image_urls(text, image_urls)

        if self.verbose:
            print(text, file=sys.stderr)


        # Evaluate prompt
        llama.reset()
        llama._ctx.kv_cache_clear()
        for type_, value in split_text:
            if type_ == "text":
                tokens = llama.tokenize(
                    value.encode("utf8"), add_bos=True, special=True
                )
                if llama.n_tokens + len(tokens) > llama.n_ctx():
                    raise ValueError(
                        f"Prompt exceeds n_ctx: {llama.n_tokens + len(tokens)} > {llama.n_ctx()}"
                    )
                llama.eval(tokens)
            else:
                image_bytes = self.load_image(value)
                embed = self._embed_image_bytes(image_bytes, llama.context_params.n_threads_batch)
                if llama.n_tokens + embed.contents.n_image_pos > llama.n_ctx():
                    raise ValueError(
                        f"Prompt exceeds n_ctx: {llama.n_tokens + embed.contents.n_image_pos} > {llama.n_ctx()}"
                    )
                n_past = ctypes.c_int(llama.n_tokens)
                n_past_p = ctypes.pointer(n_past)
                with suppress_stdout_stderr(disable=self.verbose):
                    self._llava_cpp.llava_eval_image_embed(
                        llama.ctx,
                        embed,
                        llama.n_batch,
                        n_past_p,
                    )
                # Required to avoid issues with hf tokenizer
                llama.input_ids[llama.n_tokens: n_past.value] = -1
                llama.n_tokens = n_past.value

        # Get prompt tokens to avoid a cache miss
        prompt = llama.input_ids[: llama.n_tokens].tolist()

        if response_format is not None and response_format["type"] == "json_object":
            grammar = _grammar_for_response_format(response_format)

        # Convert legacy functions to tools
        if functions is not None:
            tools = [
                {
                    "type": "function",
                    "function": function,
                }
                for function in functions
            ]

        # Convert legacy function_call to tool_choice
        if function_call is not None:
            if isinstance(function_call, str) and (
                function_call == "none" or function_call == "auto"
            ):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }

        tool = None
        if (
            tool_choice is not None
            and isinstance(tool_choice, dict)
            and tools is not None
        ):
            name = tool_choice["function"]["name"]
            tool = next((t for t in tools if t["function"]["name"] == name), None)
            if tool is None:
                raise ValueError(f"Tool choice '{name}' not found in tools.")
            schema = tool["function"]["parameters"]
            try:
                # create grammar from json schema
                grammar = llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=llama.verbose
                )
            except Exception as e:
                if llama.verbose:
                    print(str(e), file=sys.stderr)
                grammar = llama_grammar.LlamaGrammar.from_string(
                    llama_grammar.JSON_GBNF, verbose=llama.verbose
                )

        completion_or_chunks = llama.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=top_logprobs if logprobs else None,
            stream=stream,
            stop=stop,
            seed=seed,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )
        if tool is not None:
            tool_name = tool["function"]["name"]
            return _convert_completion_to_chat_function(
                tool_name, completion_or_chunks, stream
            )
        return _convert_completion_to_chat(completion_or_chunks, stream=stream)

    @staticmethod
    def _load_image(image_url: str) -> bytes:
        if image_url.startswith("file:"):
            file_path = "/" + re.sub("file:/+(.+)", "\\1", image_url)
            with Image.open(file_path) as img:
                # Optional: convert to RGB to ensure compatibility
                img = img.convert("RGB")
                buffer = BytesIO()
                img.save(buffer, format="PNG")  # or PNG
                return buffer.getvalue()
        elif image_url.startswith("data:"):
            image_bytes = base64.b64decode(image_url.split(",")[1])
            return image_bytes
        else:
            with urllib.request.urlopen(image_url) as f:
                image_bytes = f.read()
                return image_bytes

    @staticmethod
    def get_image_urls(messages: list[llama_types.ChatCompletionRequestMessage]):
        image_urls: list[str] = []
        for message in messages:
            if message["role"] == "user":
                if message["content"] is None:
                    continue
                for content in message["content"]:
                    if isinstance(content, dict) and "type" in content:
                        if content["type"] == "image":
                            if "url" in content:
                                image_urls.append(content["url"])
                            else:
                                image_urls.append(content["text"])  # assumes base64
        return image_urls

    @staticmethod
    def split_text_on_image_urls(text: str, image_urls: list[str]):
        def find_first(s: str, substrs: list[str]):
            for i, substr in enumerate(substrs):
                pos = s.find(substr)
                if pos != -1:
                    return pos, i
            return None, None

        split_text: list[tuple[Literal["text", "image_url"], str]] = []
        remaining = text
        while remaining:
            # Find first image_url
            pos, i = find_first(remaining, image_urls)
            if pos is not None and i is not None:
                if pos > 0:
                    split_text.append(("text", remaining[:pos]))
                split_text.append(("image_url", image_urls[i]))
                remaining = remaining[pos + len(image_urls[i]):]
            else:
                split_text.append(("text", remaining))
                remaining = ""
        return split_text

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str],
        local_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        **kwargs: Any,
    ) -> "Gemma3ChatHandler":
        import fnmatch
        from pathlib import Path

        try:
            from huggingface_hub import hf_hub_download, HfFileSystem  # type: ignore
            from huggingface_hub.utils import validate_repo_id  # type: ignore
        except ImportError:
            raise ImportError(
                "Llama.from_pretrained requires the huggingface-hub package. "
                "You can install it with `pip install huggingface-hub`."
            )

        validate_repo_id(repo_id)

        hffs = HfFileSystem()

        files = [
            file["name"] if isinstance(file, dict) else file
            for file in hffs.ls(repo_id)  # type: ignore
        ]

        # split each file into repo_id, subfolder, filename
        file_list: list[str] = []
        for file in files:
            rel_path = Path(file).relative_to(repo_id)
            file_list.append(str(rel_path))

        matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

        if len(matching_files) == 0:
            raise ValueError(
                f"No file found in {repo_id} that match {filename}\n\n"
                f"Available Files:\n{json.dumps(file_list)}"
            )

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple files found in {repo_id} matching {filename}\n\n"
                f"Available Files:\n{json.dumps(files)}"
            )

        (matching_file,) = matching_files

        subfolder = str(Path(matching_file).parent)
        filename = Path(matching_file).name

        # download the file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=cast(Union[str, Path, None], local_dir),
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cast(Union[str, Path, None], cache_dir),
        )

        if local_dir is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=cast(Union[str, Path, None], cache_dir),
                local_files_only=True,
            )
        else:
            model_path = os.path.join(local_dir, filename)

        return cls(
            clip_model_path=model_path,
            **kwargs,
        )
