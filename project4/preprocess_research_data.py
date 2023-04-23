import sys
import json
from pathlib import Path
import ijson
from tqdm import tqdm


def normalize_types(fp: Path) -> None:
    out_fp = fp.parent / ('norm_' + fp.name)
    if out_fp.exists():
        print('Using already normalized json file.')
        return out_fp
    with open(out_fp, 'w', encoding='utf8') as fid1:
        with open(fp, 'r', encoding='utf8') as fid2:
            for line in fid2:
                fid1.write(line.replace('NumberInt(', '').replace(')', ''))
    return out_fp


def main():
    json_path = Path(sys.argv[1])
    json_path = normalize_types(json_path)
    with open(json_path, 'rb') as fid:
        chunks_dir = Path(json_path.parent) / 'chunks'
        chunks_dir.mkdir(parents=True, exist_ok=True)
        curr_chunk_id = 0
        chunk_file = open(chunks_dir / f'chunk_{curr_chunk_id}.json', 'w', encoding='utf8')
        for i, record in tqdm(enumerate(ijson.items(fid, 'item'))):
            chunk_id = i // 250000
            if curr_chunk_id != chunk_id:
                chunk_file.close()
                curr_chunk_id = chunk_id
                chunk_file = open(chunks_dir / f'chunk_{curr_chunk_id}.json', 'w', encoding='utf8')
            chunk_file.write(json.dumps(record, default=str) + '\n')
        chunk_file.close()
        print(f'The data is ready to be loaded by spark. Check out: \'{chunks_dir.resolve()}\'')


if __name__ == '__main__':
    main()
