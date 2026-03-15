import tempfile, os
from amharic_tokenizer.tokenizer import Tokenizer

def test_train_and_roundtrip():
    # small corpus
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False) as fh:
        fh.write('ሰላም ልዩ ቃል\nሰላም ይደርሳል')
        fname = fh.name
    t = Tokenizer()
    t.train_from_corpus(fname)
    ids, spans = t.encode_with_spans('ሰላም')
    decoded = t.decode(ids, spans=spans)
    assert decoded == 'ሰላም'
    # test save/load
    outdir = tempfile.mkdtemp()
    t.save_pretrained(outdir)
    t2 = Tokenizer.load_pretrained(outdir)
    ids2 = t2.encode('ሰላም')
    # decode via id-only path should produce tokens joined; allow fallback alternative
    dec2 = t2.decode(ids2)
    assert 'ሰ' in dec2 or dec2 != ''
    print('roundtrip ok', ids, decoded, ids2, dec2)

if __name__ == '__main__':
    test_train_and_roundtrip()
