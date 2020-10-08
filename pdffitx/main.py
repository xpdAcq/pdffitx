import fire

import pdfstream.cli as cli

COMMANDS = {
    'average': cli.average,
    'integrate': cli.integrate,
    'waterfall': cli.waterfall,
    'visualize': cli.visualize,
    'instrucalib': cli.instrucalib
}


def main():
    """The CLI entry point. Run google-fire on the name - function mapping."""
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()
