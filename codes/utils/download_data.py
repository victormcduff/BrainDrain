import requests
import shutil
import os
from tqdm import tqdm
import argparse

local_nsd = "/data/ArkadiyArchive/Brain/NSA/"
url_nsd = "https://natural-scenes-dataset.s3.amazonaws.com/"

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--session_start",
		type=int,
		default=1,
		help="first session",
	)
	parser.add_argument(
		"--session_end",
		type=int,
		default=40,
		help="last session",
    )
	parser.add_argument(
		"--subject",
		type=int,
		required=True,
		default=None,
		help="subject id, 0-8",
	)
	parser.add_argument(
		"--override",
		type=int,
		default=0,
		help="override existing files?",
	)

	opt = parser.parse_args()
	subject = opt.subject
	session_start = opt.session_start
	session_end = opt.session_end
	override = opt.override

	data_directory_url = f"nsddata_betas/ppdata/subj{subject:02}/func1pt8mm/betas_fithrf_GLMdenoise_RR/" 
	local_directory = local_nsd + data_directory_url

	os.makedirs(local_directory, exist_ok=True)

	for i in range(session_start, session_end+1):
		data_name = f"betas_session{i:02}.hdf5"
		local_filename = local_directory + data_name
		url = url_nsd + data_directory_url + data_name

		if not os.path.exists(local_filename) or override: #file has not yet been downloaded
			print('Downloading file ', data_name)

			response = requests.get(url, stream=True)
			total_size_in_bytes= int(response.headers.get('content-length', 0))
			block_size = 1024 #1 KB
			progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
			with open(local_filename, 'wb') as file:
				for data in response.iter_content(block_size):
					progress_bar.update(len(data))
					file.write(data)

			progress_bar.close()

if __name__ == "__main__":
    main()
