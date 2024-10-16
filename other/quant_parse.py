from os.path import isfile
from mido import MidiFile


def quant_parse(num: int = 1, path: str = './bach/invent/invent') -> int:
    """Parses a lilypond file for the maximum note quantization. 

    Args:
        num (int): work number to analyze. Defaults to 1.
        path (str, optional): path to directory of works. Defaults to './bach/invent/invent'.

    Returns:
        int: optimal quantization value
    """

    # File check
    if not isfile(path + str(num) + '.mid'):
        raise FileNotFoundError(f'File of path {path}, #{num} does not exist.')

    # The meat of the function
    max_quant = 0
    mid_file = MidiFile(path + str(num) + '.mid')
    numerator = None; denominator = None
    for msg in mid_file:
        if msg.type == 'time_signature':
             numerator = msg.numerator
             denominator = msg.denominator
        elif msg.type == 'set_tempo':
             tempo = msg.tempo / 1_000_000
        elif msg.type in ['note_on', 'note_off'] and msg.time != 0:
            quant = int(tempo / msg.time)
            if quant & (quant - 1) == 0 and quant != 0 and quant > max_quant: # bit trick!
                max_quant = quant
    return max_quant * numerator * int(denominator / 4)


# if __name__ == '__main__':
#     q = quant_parse(1)
#     print(q)