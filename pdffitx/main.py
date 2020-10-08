import fire

import pdffitx.cli as cli

try:
    import diffpy.pdfgetx

    PDFGETX_AVAILABLE = True
    del diffpy.pdfgetx
except ImportError:
    PDFGETX_AVAILABLE = False

COMMANDS = {}
if PDFGETX_AVAILABLE:
    COMMANDS.update({"instrucalib": cli.instrucalib})


def main():
    """The CLI entry point. Run google-fire on the name - function mapping."""
    fire.Fire(COMMANDS)


if __name__ == "__main__":
    main()
