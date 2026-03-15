import argparse, os
from .tokenizer import Tokenizer
from .maps import build_unique_cv_key_maps, save_json_map

def main(argv=None):
    p = argparse.ArgumentParser(prog='amharic-tokenizer')
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--corpus', required=True, help='Base corpus file path')
    t.add_argument('--additional-corpus', nargs='*', help='Additional corpus files to merge')
    t.add_argument('--outdir', required=True)
    t.add_argument('--use-old-algorithm', action='store_true', help='Use old Suffix Automaton algorithm instead of new Greedy Meaningful Subword')

    enc = sub.add_parser('encode')
    enc.add_argument('--tokenizer', required=True)
    enc.add_argument('--text', required=True)

    dec = sub.add_parser('decode')
    dec.add_argument('--tokenizer', required=True)
    dec.add_argument('--ids', nargs='+', type=int, required=True)

    maps = sub.add_parser('build-maps')
    maps.add_argument('--outdir', required=True)

    args = p.parse_args(argv)

    if args.cmd == 'train':
        tok = Tokenizer()
        
        # Determine which algorithm to use (default is new algorithm)
        use_new_algorithm = not args.use_old_algorithm
        
        # Check if additional corpora are provided
        if args.additional_corpus:
            # Train from multiple corpora
            all_corpora = [args.corpus] + args.additional_corpus
            tok.train_from_multiple_corpora(all_corpora, use_new_algorithm=use_new_algorithm)
        else:
            # Train from single corpus (original behavior)
            tok.train_from_corpus(args.corpus, use_new_algorithm=use_new_algorithm)
        
        algorithm_name = "OLD (Suffix Automaton)" if args.use_old_algorithm else "NEW (Greedy Meaningful Subword)"
        print(f'Trained using {algorithm_name} algorithm')
        tok.save_pretrained(args.outdir)
        key_to_glyph, glyph_to_key = build_unique_cv_key_maps()
        save_json_map(key_to_glyph, os.path.join(args.outdir, 'key_to_glyph.json'))
        save_json_map(glyph_to_key, os.path.join(args.outdir, 'glyph_to_key.json'))
        print('Saved tokenizer and maps to', args.outdir)
    elif args.cmd == 'encode':
        tok = Tokenizer.load_pretrained(os.path.dirname(args.tokenizer) if os.path.isdir(args.tokenizer) else os.path.dirname(args.tokenizer))
        ids = tok.encode(args.text)
        print(' '.join(map(str, ids)))
    elif args.cmd == 'decode':
        tok = Tokenizer.load_pretrained(os.path.dirname(args.tokenizer) if os.path.isdir(args.tokenizer) else os.path.dirname(args.tokenizer))
        text = tok.decode(args.ids)
        print(text)
    elif args.cmd == 'build-maps':
        key_to_glyph, glyph_to_key = build_unique_cv_key_maps()
        save_json_map(key_to_glyph, os.path.join(args.outdir, 'key_to_glyph.json'))
        save_json_map(glyph_to_key, os.path.join(args.outdir, 'glyph_to_key.json'))
        print('Saved maps to', args.outdir)
    else:
        p.print_help()
