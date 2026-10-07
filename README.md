# Uzbek STT

O'zbekcha nutqni matnga o'giruvchi veb-bot. Mikrofondan yozib olish yoki audio fayl yuklash orqali ishlaydi.

## Talablar

- Python 3.12
- NVIDIA GPU + drayver (CUDA). VRAM sarfi ~1.6–2GB — 4GB'lik videokartaga ham sig'adi.
- `ffmpeg` (audio formatlarini decode qilish uchun)

```bash
sudo apt install ffmpeg
```

## O'rnatish

```bash
cd uzbek_stt
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`model/` papkasi loyiha ichida tayyor holda keladi — internetdan hech narsa yuklab olish shart emas.

## Ishga tushirish

```bash
./venv/bin/uvicorn app:app --host 0.0.0.0 --port 8111
```

Brauzerda ochish: `http://localhost:8111`

Fonda (orqa fonda) ishga tushirish:

```bash
setsid ./venv/bin/uvicorn app:app --host 0.0.0.0 --port 8111 > server.log 2>&1 < /dev/null &
disown
```

## Boshqa kompyuterga ko'chirish

Butun `uzbek_stt/` papkasini (model bilan birga) ko'chiring, so'ng o'sha kompyuterda:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn app:app --host 0.0.0.0 --port 8111
```

`venv/` papkasini ko'chirmang — uni yangi joyda qayta yarating, chunki torch/CUDA versiyalari kompyuterga bog'liq.

## Sozlash

Model yo'lini o'zgartirish uchun muhit o'zgaruvchisi:

```bash
export STT_MODEL_PATH=/boshqa/yo'l/model
```

## Muammolarni bartaraf etish

- **`ffmpeg` topilmadi** — audio decode qilishda xatolik beradi, `sudo apt install ffmpeg` bilan o'rnatiladi.
- **CUDA topilmadi** — model avtomatik CPU'da ishlaydi (sekinroq, lekin ishlaydi).
- **Port band** — `--port` qiymatini boshqasiga almashtiring.
