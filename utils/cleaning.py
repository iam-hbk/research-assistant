

def remove_emojis(text : str ) -> str:
    '''remove imojis from texts string'''
    try:
        if text: 
            text = emoji.replace_emoji(text, '')
            text = ' '.join(text.split())
            return text
        else: return None
    except Exception as e:
        print(e)
        return None