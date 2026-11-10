import csv, io, json, os, subprocess, time, zipfile, shutil
from pathlib import Path
from urllib.parse import quote
root = Path('/work/users/p/m/pmucharl/data/SEED_IV')
root.mkdir(parents=True, exist_ok=True)
cli = str(Path.home()/'.local/bin/kaggle')
os.environ.update(KAGGLE_USERNAME='anonymous', KAGGLE_KEY='anonymous')
listing = subprocess.check_output([cli,'datasets','files','phhasian0710/seed-iv','--page-size','100','--csv'], text=True)
rows = list(csv.DictReader(io.StringIO(listing[listing.index('name,size,creationDate'):])) )
raw = [r for r in rows if r['name'].startswith('eeg_raw_data/')]
assert len(raw) == 45, len(raw)
assert all(sum(r['name'].startswith('eeg_raw_data/'+s+'/') for r in raw)==15 for s in ('1','2','3'))
(root/'raw_manifest.json').write_text(json.dumps(raw, indent=2))
for i,r in enumerate(raw,1):
    target = root/r['name']; target.parent.mkdir(parents=True,exist_ok=True)
    encoded = quote(r['name'], safe='')
    downloaded = target.parent/encoded
    if not (target.exists() and target.stat().st_size==int(r['size'])):
        if not downloaded.exists():
            subprocess.run([cli,'datasets','download','phhasian0710/seed-iv','-f',encoded,'-p',str(target.parent),'-q'],check=True)
        if zipfile.is_zipfile(downloaded):
            with zipfile.ZipFile(downloaded) as archive:
                entries=archive.infolist()
                assert len(entries)==1 and entries[0].file_size==int(r['size'])
                with archive.open(entries[0]) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
            assert target.stat().st_size==int(r['size'])
            downloaded.unlink()
        else:
            assert downloaded.stat().st_size==int(r['size'])
            downloaded.rename(target)
    print(f"VERIFIED {i}/45 {r['name']} {target.stat().st_size}",flush=True)
files=list((root/'eeg_raw_data').glob('*/*.mat'))
assert len(files)==45
summary={'status':'complete','files':45,'bytes':sum(p.stat().st_size for p in files),'source':'phhasian0710/seed-iv','subset':'eeg_raw_data'}
(root/'download_complete.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
