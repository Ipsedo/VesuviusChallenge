# -*- coding: utf-8 -*-
import argparse


def main() -> None:
    parser = argparse.ArgumentParser("vesuvius_challenge main")

    parser.add_argument("message", type=str, help="The message to display")

    args = parser.parse_args()

    print(args)


if __name__ == "__main__":
    main()
