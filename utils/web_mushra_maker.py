import os
import yaml
from glob import glob
import argparse


def create_trial(dir_name):
    clip_paths = glob(dir_name + "/*.wav")
    files = [x.split('/')[-1] for x in clip_paths]
    files.remove('ref.wav')
    dir = dir_name.split('/', 1)[1]

    trial = f"""
          - type: mushra
            id: {dir.rsplit('/', 1)[-1]}
            name: MUSHRA
            content: {dir.rsplit('/', 1)[-1]}, rate each of the test items based on how accurately they approximate the timbre of the reference
            enableLooping: true 
            reference: {dir}/ref.wav
            showConditionNames: false
            createAnchor35: false
            createAnchor70: false
            stimuli:
"""

    for n, i in enumerate(files):
        trial += f"""                  C{i.split('.')[0]}: {dir}/{i}
"""

    return trial




def main(directory, output_file):

    # Example usage

    trial_list = glob(f'{directory}/*')

    anchor = 'false'
    show_cond = 'false'

    newf = f"""testname: Pilot Test
testId: Pilot
bufferSize: 2048
stopOnErrors: true
showButtonPreviousPage: true
remoteService: service/write.php


pages:
    - type: generic
      id: first_page
      name: Welcome
      content: Welcome to a listening test!
    - type: mushra
      id: Training 1
      name: Training 1
      content: This is the first training page, rate each of the test items based on how accurately they approximate the timbre of the reference
      showWaveform: true
      enableLooping: true
      reference: configs/resources/audio/training_trials/train_1/ref.wav
      createAnchor35: {anchor}
      createAnchor70: {anchor}
      showConditionNames: true
      stimuli:
          C1: configs/resources/audio/training_trials/train_1/RealTimeDeepLearning-devHT5-WaveNet3-6.wav
          C2: configs/resources/audio/training_trials/train_1/RealTimeDeepLearning-devHT5-RNN96-6.wav
          C3: configs/resources/audio/training_trials/train_1/RealTimeDeepLearning-devHT5-MLP-6.wav
    - type: mushra
      id: Training 2
      name: Training 2
      content: This is the second training page, rate each of the test items based on how accurately they approximate the timbre of the reference
      showWaveform: true
      enableLooping: true
      reference: configs/resources/audio/training_trials/train_2/ref.wav
      createAnchor35: {anchor}
      createAnchor70: {anchor}
      showConditionNames: true
      stimuli:
          C1: configs/resources/audio/training_trials/train_2/GANAMP-devClean-MultiSpecCritmel7-1.wav
          C2: configs/resources/audio/training_trials/train_2/GANAMP-devClean-MultiSpecCritspec3-1.wav
          C3: configs/resources/audio/training_trials/train_2/GANAMP-devClean-MultiSpecCritspec4-1.wav
    -
          - random"""

    endf = """    - type: finish
      name: Thank you
      content: Thank you for attending!
      showResults: true
      writeResults: true
      questionnaire:
          - type: text
            label: eMail
            name: email
          - type: number
            label: Age
            name: age
            min: 0
            max: 100
            default: 30
          - type: likert
            name: gender
            label: Gender
            response:
             - value: female
               label: Female
             - value: male
               label: Male
             - value: other
               label: Other"""

    output = newf

    for each in trial_list:
        if os.path.isdir(each):
            output += create_trial(each)

    output += endf

    with open(f'{output_file}.yaml', 'w') as ff:
        ff.write(output)

