"""切块命令入口，复用原有 azusa_rag.main 流程。"""

import argparse

from azusa_rag.main import main as build_chunks


def main(argv: list[str] | None = None) -> None:
    """校验命令参数并切块；--help 不读取文档或生成产物。"""
    argparse.ArgumentParser(description="Build chunk artifacts").parse_args(argv)
    build_chunks()


if __name__ == "__main__":
    main()
