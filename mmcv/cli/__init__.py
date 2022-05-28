import click

from .ext import ext


@click.group()
def main():
    pass


main.add_command(ext)
