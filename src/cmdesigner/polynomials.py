import sys
import argparse
from typing import List
from cmdesigner.cm_synthesis import cli

def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(prog=sys.argv[0],
        usage='%(prog)s [options]',
        description='Characteristic polynomial synthesis tool',
        prefix_chars='@'
    )

    parser.add_argument('@o',
                        '@@order',
                        action='store',
                        default=4,
                        type=int,
                        help='Order of the filter. default: 5'

    )
    parser.add_argument('@z',
                        '@@zeros',
                        action='store',
                        nargs='*',
                        type=complex,
                        help='Unordered list of prescribbed finite transmission zeros. default: 1.52 1.97'
                    )
    
    parser.add_argument('@r',
                        '@@return-loss',
                        action='store',
                        default=20,
                        type=float,
                        help='Desired in-band return loss level. default:20'
                    )
    args = parser.parse_args(argv)
    print(args)
    
    cli(args)

if __name__ == '__main__':
    exit(main())