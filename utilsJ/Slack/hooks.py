import requests
import json
import os

"""
deeply inspired on abhisheck thakur 's youtube vid
intended to automatically post messages to slack channels.
webhooks should be in enviromental vars
"""


class msger():
    def __init__(self):
        try:
            # in a future just list all env vars for those containing slack in keys or https://hooks[...] in items
            self.hooks = {
                'COM' : os.environ.get('SLACK_COM'),
                'CNN' : os.environ.get('SLACK_CNN')
            }
        except:
            pint(f'coud not find hooks in envr vars named SLACK_COM and/or SLACK_CNN')

    def msg(self, channel, message):
        """
        send message to desired channel .msg('channel', 'bodytext')
        """
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
