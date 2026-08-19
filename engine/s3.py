"""Minimal S3 get/put — the image ships boto3, not the aws CLI.

    python engine/s3.py get s3://bucket/key /local/path
    python engine/s3.py put /local/path s3://bucket/key

Credentials come from the environment. NOTE: the prod EC2 instance role currently
cannot read s3://xapien-vton-engines — pass explicit keys for now (docker/DEPLOY.md).
"""
import sys
import urllib.parse

import boto3


def _split(uri: str) -> tuple[str, str]:
    u = urllib.parse.urlparse(uri)
    if u.scheme != "s3":
        raise SystemExit(f"not an s3:// uri: {uri}")
    return u.netloc, u.path.lstrip("/")


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("get", "put"):
        raise SystemExit(__doc__)
    cmd, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    s3 = boto3.client("s3")
    if cmd == "get":
        bucket, key = _split(a)
        s3.download_file(bucket, key, b)
        print(f"got  {a} -> {b}")
    else:
        bucket, key = _split(b)
        s3.upload_file(a, bucket, key)
        print(f"put  {a} -> {b}")


if __name__ == "__main__":
    main()
