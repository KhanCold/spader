# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from contextlib import redirect_stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Pyserini prebuilt BM25 (Lucene) index into a user-specified directory. "
            "This avoids using ~/.cache/pyserini so it is easier to manage with project-local data/." 
        )
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="wikipedia-dpr",
        help=(
            "Pyserini prebuilt index key (TF_INDEX_INFO). Example: wikipedia-dpr, enwiki-paragraphs, "
            "wiki-all-6-3-tamber. Default: wikipedia-dpr."
        ),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/pyserini_indexes",
        help="Directory to download + unpack into (default: ./data/pyserini_indexes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the index directory already exists.",
    )

    args = parser.parse_args()

    from pyserini.prebuilt_index_info import TF_INDEX_INFO
    from pyserini.util import download_and_unpack_index

    if args.index_name not in TF_INDEX_INFO:
        raise ValueError(
            f"Unknown index_name={args.index_name!r}. "
            f"This script only supports TF_INDEX_INFO keys."
        )

    target = TF_INDEX_INFO[args.index_name]
    urls = target.get("urls") or []
    if not urls:
        raise ValueError(f"No download URL found for index_name={args.index_name!r}.")

    os.makedirs(args.save_path, exist_ok=True)

    last_err: Exception | None = None
    for url in urls:
        try:
            # pyserini prints progress logs to stdout; redirect those logs to stderr so
            # stdout can be safely captured as the returned path in shell scripts.
            with redirect_stdout(sys.stderr):
                index_path = download_and_unpack_index(
                    url,
                    index_directory=args.save_path,
                    local_filename=target.get("filename", False),
                    force=args.force,
                    verbose=True,
                    prebuilt=False,
                    md5=target.get("md5"),
                )
            print(index_path)
            return
        except Exception as e:  # pragma: no cover
            last_err = e
            print(f"Unable to download index at {url}, trying next URL... ({type(e).__name__}: {e})")

    raise RuntimeError(f"Unable to download index {args.index_name!r} from any known URLs.") from last_err


if __name__ == "__main__":
    main()
