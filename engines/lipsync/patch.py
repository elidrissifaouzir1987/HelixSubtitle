"""Patche le dépôt Wav2Lip cloné pour fonctionner avec des versions récentes
(librosa/numpy) et tolérer les frames sans visage. Idempotent."""
from pathlib import Path

W2L = Path(__file__).resolve().parent / "Wav2Lip"


def patch(path: Path, old: str, new: str) -> None:
    txt = path.read_text(encoding="utf-8")
    if new in txt:
        print(f"  déjà patché : {path.name}")
        return
    if old not in txt:
        print(f"  MOTIF INTROUVABLE dans {path.name} (déjà modifié ?)")
        return
    path.write_text(txt.replace(old, new), encoding="utf-8")
    print(f"  patché : {path.name}")


# 1) librosa.filters.mel : args positionnels -> nommés (librosa >= 0.10)
patch(W2L / "audio.py",
      "librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
      "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,")

# 2) np.int retiré dans numpy >= 1.24
patch(W2L / "face_detection" / "utils.py",
      "newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)",
      "newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=int)")

# 3) tolérer les frames sans visage : réutiliser la dernière détection
OLD = """	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])"""
NEW = """	results = []
	pady1, pady2, padx1, padx2 = args.pads
	last = None
	for rect, image in zip(predictions, images):
		if rect is None:
			if last is not None:
				results.append(last); continue
			results.append([0, 0, image.shape[1], image.shape[0]]); continue
		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		last = [x1, y1, x2, y2]
		results.append(last)
	if all(r == [0, 0, images[0].shape[1], images[0].shape[0]] for r in results):
		raise ValueError('Aucun visage detecte dans la video.')"""
patch(W2L / "inference.py", OLD, NEW)
print("Patch terminé.")
