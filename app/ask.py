"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 初回質問を省略すると、講座に固有の質問をする

ツールを持たない「素の」呼び出しだが、1回の実行中は会話履歴を保つ。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None) -> str:
    """質問を送り、指定された会話履歴に質問と回答を追加する。"""
    client = genai.Client(api_key=get_api_key())
    conversation = history if history is not None else []
    user_message = types.Content(
        role="user", parts=[types.Part.from_text(text=question)]
    )
    response = client.models.generate_content(
        model=MODEL, contents=[*conversation, user_message]
    )

    usage = response.usage_metadata
    print(f"[入力トークン数] {usage.prompt_token_count}")
    print(f"[出力トークン数] {usage.candidates_token_count}")

    answer = response.text
    conversation.extend(
        [user_message, types.Content(role="model", parts=[types.Part.from_text(text=answer)])]
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    history: list[types.Content] = []
    try:
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            print(ask(question, history))
            try:
                question = input("\n続けて質問（終了するには空行または Ctrl-D）: ").strip()
            except EOFError:
                print()
                break
            if not question:
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
