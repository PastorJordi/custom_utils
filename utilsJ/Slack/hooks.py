import requests
import json
import os
import slack

"""
expanding functionallity to a bot, so we can directly wisper to users, hence reducing flood.
Never used it previously, though

more info: https://github.com/slackapi/python-slackclient
"""
class whisp():
    """ to avoid flood, now bot will just whisper me (ie DM), use method whisper() """
    def __init__(self):
        try:
            self.slack_bot_token = os.environ['SLACK_BOT_TOKEN']
        except:
            pint(f'coud not find bot token as "SLACK_BOT_TOKEN" in env vars')

        self.contacts = {
            'jordi' : 'U8J8YA66S',
            'jaime' : 'U7UTKNN0P',
            'lejla' : 'U7TTEEN4T',
            'genis' : 'U7U0GRABF'
        }

    def whisper(self, dst, text):
        """dst: who [in contacts], text= message"""
        try:
            data = {
                'token': self.slack_bot_token,
                'channel': self.contacts[dst],
                'as_user': True,
                'text': text
            }

            requests.post(url='https://slack.com/api/chat.postMessage',
                        data=data)
        except:
            print(f'couldnt whisper that, probably user is not in contact list ({self.contacts.keys()})')












"""
deeply inspired on abhisheck thakur 's youtube vid
intended to automatically post messages to slack channels.
webhooks should be in enviromental vars
"""


class msger():
    def __init__(self):
        
        self.corresponding_channels = {
            'COM' : '#changes_of_mind',
            'CNN' : '#biases_poses_cnn'
        }
        self.contacts = {
            'jordi' : 'U8J8YA66S',
            'jaime' : 'U7UTKNN0P',
            'lejla' : 'U7TTEEN4T',
            'genis' : 'U7U0GRABF'
        }

    def msg(self, channel, message, uploadpath=None):
        """everything is string based
        importantly, channelname are RAW ie: "#changes_of_mind" """
        if channel in self.contacts.keys():
            channel = self.contacts[channel]

        try:
            #client = slack.WebClient(token=os.environ['SLACK_API_TOKEN'])
            client = slack.WebClient(token=os.environ['SLACK_BOT_TOKEN'])
        except:
            print('could not auth api token')
            return 'failure'

        if uploadpath is not None:
            # do stuff
            if os.path.exists(uploadpath):
                
                response = client.files_upload(
                        channels=channel,
                        file=uploadpath,
                        initial_comment=message)
            else:
                print(f'could not find {uploadpath}')
        else:
            response = client.chat_postMessage(
                    channel=channel,
                    text=message)

    def webmsg(self, channel, message):
        """
        send message to desired channel .msg('channel', 'bodytext')
        everything work with strings
        """
        try:
            # in a future just list all env vars for those containing slack in keys or https://hooks[...] in items
            self.hooks = {
                'COM' : os.environ.get('SLACK_COM'),
                'CNN' : os.environ.get('SLACK_CNN')
            }
        except:
            pint(f'coud not find hooks in envr vars named SLACK_COM and/or SLACK_CNN')


            if channel in self.hooks.keys():
                data = {
                    'text': message
                }
                requests.post(
                    self.hooks[channel],
                    json.dumps(data)
                )
            else:
                print(f'{channel} not in available ones ({list(self.hooks.keys())})')
                print('try again')

