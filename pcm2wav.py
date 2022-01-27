import wave
import soundfile as sf
import os

# pcm format 파일에 대한 wav 파일을 생성하고, 주소를 반환한다. 
def pcm2wav( pcm_file, channels=1, bit_depth=16, sampling_rate=16000 ):
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth "+str(bit_depth)+" must be a multiple of 8.")

    wav_file = pcm_file[:-3]+"wav"  
    if not os.path.isfile(wav_file):
        # Read the .pcm file as a binary file and store the data to pcm_data
        with open( pcm_file, 'rb') as opened_pcm_file:
            pcm_data = opened_pcm_file.read()
            obj2write = wave.open( wav_file, 'wb')
            obj2write.setnchannels( channels )
            obj2write.setsampwidth( bit_depth // 8 )
            obj2write.setframerate( sampling_rate )
            obj2write.writeframes( pcm_data )
            obj2write.close()
    return wav_file
