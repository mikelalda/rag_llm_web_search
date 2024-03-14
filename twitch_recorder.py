import datetime
import enum
import getopt
import logging
import os
import subprocess
import sys
import shutil
import time
from streamlink import Streamlink
import requests
import cv2
import config
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import ffmpeg
import csv
from scipy.io import wavfile 



class TwitchResponseStatus(enum.Enum):
    ONLINE = 0
    OFFLINE = 1
    NOT_FOUND = 2
    UNAUTHORIZED = 3
    ERROR = 4
    AUTORIZED = 5
    NO_VIDEOS = 6


class TwitchRecorder:
    def __init__(self):
        # global configuration
        self.ffmpeg_path = "ffmpeg"
        self.disable_ffmpeg = False
        self.refresh = 15
        self.root_path = config.root_path

        # user configuration
        self.username = config.username
        self.quality = "audio" #160p (worst), 360p, 480p, 720p, 720p60, 1080p60 (best)

        # twitch configuration
        self.client_id = config.client_id
        self.client_secret = config.client_secret
        self.token_url = "https://id.twitch.tv/oauth2/token?client_id=" + self.client_id + "&client_secret=" \
                         + self.client_secret + "&grant_type=client_credentials"
        self.url = "https://api.twitch.tv/helix/streams"
        self.video_url = "https://api.twitch.tv/helix/videos"
        self.url_id = "https://api.twitch.tv/helix/users"
        self.access_token = self.fetch_access_token()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "distil-whisper/distil-large-v2"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

    def fetch_access_token(self):
        token_response = requests.post(self.token_url, timeout=15)
        token_response.raise_for_status()
        token = token_response.json()
        return token["access_token"]

    def save_previous_videos(self):
        # path to recorded stream
        recorded_path = os.path.join(self.root_path, "recorded", self.username)
        # path to finished video, errors removed
        processed_path = os.path.join(self.root_path, "processed", self.username)

        # create directory for recordedPath and processedPath if not exist
        if os.path.isdir(recorded_path) is False:
            os.makedirs(recorded_path)
        if os.path.isdir(processed_path) is False:
            os.makedirs(processed_path)

        # fix videos from previous recording session
        try:
            self.videos_download(recorded_path, processed_path)
        except Exception as e:
            logging.error(e)
            logging.info("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))

    def run(self):
        # path to recorded stream
        recorded_path = os.path.join(self.root_path, "data/recorded", self.username)
        # path to finished video, errors removed
        processed_path = os.path.join(self.root_path, "data/processed", self.username)

        # create directory for recordedPath and processedPath if not exist
        if os.path.isdir(recorded_path) is False:
            os.makedirs(recorded_path)
        if os.path.isdir(processed_path) is False:
            os.makedirs(processed_path)

        # make sure the interval to check user availability is not less than 15 seconds
        if self.refresh < 15:
            logging.warning("check interval should not be lower than 15 seconds")
            self.refresh = 15
            logging.info("system set check interval to 15 seconds")

        # fix videos from previous recording session
        try:
            video_list = [f for f in os.listdir(recorded_path) if os.path.isfile(os.path.join(recorded_path, f))]
            if len(video_list) > 0:
                logging.info("processing previously recorded files")
            for f in video_list:
                recorded_filename = os.path.join(recorded_path, f)
                processed_filename = os.path.join(processed_path, f)
                self.process_recorded_file(recorded_filename, processed_filename)
        except Exception as e:
            logging.error(e)

        logging.info("checking for %s every %s seconds, recording with %s quality",
                     self.username, self.refresh, self.quality)
        self.loop_check(recorded_path, processed_path)

    def process_recorded_file(self, recorded_filename, processed_filename):
        if self.disable_ffmpeg:
            logging.info("moving: %s", recorded_filename)
            shutil.move(recorded_filename, processed_filename)
        else:
            logging.info("fixing %s", recorded_filename)
            self.ffmpeg_copy_and_fix_errors(recorded_filename, processed_filename)

    def ffmpeg_copy_and_fix_errors(self, recorded_filename, processed_filename):
        try:
            subprocess.call(
                [self.ffmpeg_path, "-err_detect", "ignore_err", "-i", recorded_filename, "-c", "copy",
                 processed_filename])
            os.remove(recorded_filename)
        except Exception as e:
            logging.error(e)

    def check_user(self):
        info = None
        user_id = None
        status = TwitchResponseStatus.ERROR
        try:
            headers = {"Client-ID": self.client_id, "Authorization": "Bearer " + self.access_token}
            r = requests.get(self.url + "?user_login=" + self.username, headers=headers, timeout=15)
            r.raise_for_status()
            info = r.json()
            if info is None or not info["data"]:
                status = TwitchResponseStatus.OFFLINE
            else:
                status = TwitchResponseStatus.ONLINE
        except requests.exceptions.RequestException as e:
            if e.response:
                if e.response.status_code == 401:
                    status = TwitchResponseStatus.UNAUTHORIZED
                if e.response.status_code == 404:
                    status = TwitchResponseStatus.NOT_FOUND
        return status, info, user_id
    def get_user_id(self):
        headers = {"Client-ID": self.client_id, "Authorization": "Bearer " + self.access_token}
        r = requests.get(self.url_id + "?login=" + self.username, headers=headers, timeout=15)
        r.raise_for_status()
        info = r.json()
        self.userid = info['data'][0]['id']
    def check_videos(self):
        info = None
        status = TwitchResponseStatus.ERROR
        self.get_user_id()
        try:
            headers = {"Client-ID": self.client_id, "Authorization": "Bearer " + self.access_token}
            r = requests.get(self.video_url + "?user_id=" + self.userid, headers=headers, timeout=15)
            r.raise_for_status()
            info = r.json()
            if info is None or not info["data"]:
                status = TwitchResponseStatus.NO_VIDEOS
            else:
                status = TwitchResponseStatus.AUTORIZED
        except requests.exceptions.RequestException as e:
            if e.response:
                if e.response.status_code == 401:
                    status = TwitchResponseStatus.UNAUTHORIZED
                if e.response.status_code == 404:
                    status = TwitchResponseStatus.NOT_FOUND
        return status, info
    
    def videos_download(self, recorded_path, processed_path):
        status, info = self.check_videos()
        if status == TwitchResponseStatus.NOT_FOUND:
            logging.error("username not found, invalid username or typo")
            time.sleep(self.refresh)
        elif status == TwitchResponseStatus.ERROR:
            logging.error("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))
        elif status == TwitchResponseStatus.NO_VIDEOS:
            logging.info("%s currently has no videos", self.username)
        elif status == TwitchResponseStatus.UNAUTHORIZED:
            logging.info("unauthorized. App will close")
        elif status == TwitchResponseStatus.AUTORIZED:
            logging.info("%s autorized for download", self.username)
            csv_name = 'data/' + self.username + '_twitch.csv'
            exists = os.path.isfile(csv_name)
            with open(csv_name, 'a', newline='') as results:
                fieldnames = ['title', 'filename', 'url', 'text']
                writer = csv.DictWriter(results, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                channels = info["data"]
                for channel in channels:
                    filename = channel.get("title").replace(" ", "_") + ".wav"

                    # clean filename from unnecessary characters
                    filename = "".join(x for x in filename if x.isalnum() or x in [" ", "-", "_", "."])

                    recorded_filename = os.path.join(recorded_path, filename)
                    processed_filename = os.path.join(processed_path, filename)

                    # start streamlink process
                    url = channel.get("url")
                    streamlink = Streamlink()
                    stream_url  = streamlink.streams(url)[self.quality].to_url()
                    stream = ffmpeg.input(stream_url)
                    stream = ffmpeg.output(stream, recorded_filename)
                    os.remove(recorded_filename) if os.path.exists(recorded_filename) else None
                    None if os.path.exists(processed_filename) else ffmpeg.run(stream)
                    results = None if os.path.exists(processed_filename) else self.pipe(recorded_filename)
                    if results != None:
                        data = {'text':results['text'], 'title':channel.get("title"), 'filename':filename, 'url':url}
                        [writer.writerow({k:v.encode('utf8') for k,v in data.items()})]
                         
                    logging.info("recording stream is done for %s, processing video file",filename)
                    if os.path.exists(recorded_filename) is True:
                        self.process_recorded_file(recorded_filename, processed_filename)
                    else:
                        logging.info("skip fixing, file not found")

                    logging.info("processing is done for {}".format(filename))
                
                    

    def loop_check(self, recorded_path, processed_path):
        while True:
            status, info = self.check_user()
            if status == TwitchResponseStatus.NOT_FOUND:
                logging.error("username not found, invalid username or typo")
                time.sleep(self.refresh)
            elif status == TwitchResponseStatus.ERROR:
                logging.error("%s unexpected error. will try again in 5 minutes",
                              datetime.datetime.now().strftime("%Hh%Mm%Ss"))
                time.sleep(300)
            elif status == TwitchResponseStatus.OFFLINE:
                logging.info("%s currently offline, checking again in %s seconds", self.username, self.refresh)
                time.sleep(self.refresh)
            elif status == TwitchResponseStatus.UNAUTHORIZED:
                logging.info("unauthorized, will attempt to log back in immediately")
                self.access_token = self.fetch_access_token()
            elif status == TwitchResponseStatus.ONLINE:
                logging.info("%s online, stream recording in session", self.username)

                channels = info["data"]
                channel = next(iter(channels), None)
                filename = channel.get("title").replace(" ", "_") + ".mp4"

                # clean filename from unnecessary characters
                filename = "".join(x for x in filename if x.isalnum() or x in [" ", "-", "_", "."])

                recorded_filename = os.path.join(recorded_path, filename)
                processed_filename = os.path.join(processed_path, filename)

                # start streamlink process
                subprocess.call(
                    ["streamlink", "--twitch-disable-ads", "twitch.tv/" + self.username, self.quality,
                     "-o", recorded_filename])

                logging.info("recording stream is done, processing video file")
                if os.path.exists(recorded_filename) is True:
                    self.process_recorded_file(recorded_filename, processed_filename)
                else:
                    logging.info("skip fixing, file not found")

                logging.info("processing is done, going back to checking...")
                time.sleep(self.refresh)


def main(argv):
    twitch_recorder = TwitchRecorder()
    usage_message = "twitch-recorder.py -u <username> -q <quality>"
    logging.basicConfig(filename="logs/twitch-recorder.log", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    try:
        opts, args = getopt.getopt(argv, "hu:q:l:", ["username=", "quality=", "log=", "logging=", "disable-ffmpeg"])
    except getopt.GetoptError:
        print(usage_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(usage_message)
            sys.exit()
        elif opt in ("-u", "--username"):
            twitch_recorder.username = arg
        elif opt in ("-q", "--quality"):
            twitch_recorder.quality = arg
        elif opt in ("-l", "--log", "--logging"):
            logging_level = getattr(logging, arg.upper(), None)
            if not isinstance(logging_level, int):
                raise ValueError("invalid log level: %s" % logging_level)
            logging.basicConfig(level=logging_level)
            logging.info("logging configured to %s", arg.upper())
        elif opt == "--disable-ffmpeg":
            twitch_recorder.disable_ffmpeg = True
            logging.info("ffmpeg disabled")
    twitch_recorder.save_previous_videos()
    # twitch_recorder.run()


if __name__ == "__main__":
    main(sys.argv[1:])