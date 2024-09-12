#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-

import numpy
from math import floor
from subprocess import Popen, PIPE
import pty
import os
import socket
import argparse

# Return a unicode blocks bar
def ubar( ratio, width=20):
    # Get the full bar part
    d = float( width) * ratio
    s = '█' * int( floor( d))

    # Add fractional bar part
    f = [' ','▏','▎','▍','▌','▋','▊','▉']
    r = int( floor( len(f)*(d - floor( d))))
    s += f[r]

    # Add idle time and return
    s += ' ' * int( width - floor(d) - 1)
    return s + '|'


# Return a unicode blocks stacked bar
def sbar( ratio1, ratio2, width=20):
    # Get the first string (coarse resolution)
    s = '\033[36m' + '█'*int( floor( width * ratio1)) + '\033[0m'

    # Stack the second string (fine resolution)
    d = width * ratio2
    s += '\033[91m'
    s += '█' * int( floor( d))
    f = [' ','▏','▎','▍','▌','▋','▊','▉']
    r = int( floor( len(f)*(d - floor( d))))
    s += f[r] + '\033[0m'

    # Add the idle time string and return
    s += ' '*int( width - floor( width * ratio1) - floor(d) - 1)
    return s + '\033[36m|\033[0m'


# Return a unicode blocks notched bar
def nbar( ratio1, ratio2, width=20):
    # Get the compositions
    if ratio1 > ratio2:
        c1 = int( floor( (width * ratio2)))
        c2 = int( floor( (width * (ratio1-ratio2))))
        c3 = 0
    else:
        c1 = int( floor( (width * ratio1)))
        c2 = 0
        c3 = int( floor( (width * (ratio2-ratio1))))

    # Make bar string
    c4 = width-c1-c2-c3
    return '█'*c1 + '▊' + '█'*(c2+c3) + ' '*c4 + '|'


# Return a unicode blocks dual bar
def dbar( ratio1, ratio2, width=20):
    # Building blocks
    b  = ['█','▀','▄',' ']

    # Get the compositions
    if ratio1 > ratio2:
        c1 = int( floor( (width * ratio2)))
        c2 = int( floor( (width * (ratio1-ratio2))))
        c3 = 0
    else:
        c1 = int( floor( (width * ratio1)))
        c2 = 0
        c3 = int( floor( (width * (ratio2-ratio1))))

    # Make bar string
    c4 = width-c1-c2-c3
    return b[0]*c1 + b[1]*c2 + b[2]*c3 + b[3]*c4 + '|'


# Return a unicode Braille quad bar
_b = ['⠉','⠒','⠤','⣀','⠛','⠭','⣉','⠶','⣒','⣤','⠿','⣛','⣭','⣶','⣿',' ']
_B = numpy.empty( (2,2,2,2), dtype=numpy.str_)

_B[1,0,0,0] = _b[0]
_B[0,1,0,0] = _b[1]
_B[0,0,1,0] = _b[2]
_B[0,0,0,1] = _b[3]

_B[1,1,0,0] = _b[4]
_B[1,0,1,0] = _b[5]
_B[1,0,0,1] = _b[6]
_B[0,1,1,0] = _b[7]
_B[0,1,0,1] = _b[8]
_B[0,0,1,1] = _b[9]

_B[1,1,1,0] = _b[10]
_B[1,1,0,1] = _b[11]
_B[1,0,1,1] = _b[12]
_B[0,1,1,1] = _b[13]

_B[1,1,1,1] = _b[14]
_B[0,0,0,0] = _b[15]

def qbar( r, wid=20):
    if len( r) == 2:
        r = [r[0],0,r[1],0]
    t = [[int(v/wid < r[i]) for v in range( wid)] for i in range( len( r))]
    s = ''
    for i in range( wid):
        s += _B[t[0][i],t[1][i],t[2][i],t[3][i]]
    return '|' + s + '|'

# Replacement for python 2 command.getoutput
def getoutput( c):
	import subprocess
	p = subprocess.Popen( c, stdout=subprocess.PIPE, shell=True)
	out,err = p.communicate()
	return out.decode()

# Draw the results
def draw( D, wid=20):
    global last_cpu_poll
    global host_name
    global disk_names
    global disk_sector_sizes
    global last_disk_poll

    # Remove previous output
    #print( (3*(len(D)+2)+5+len(disk_names))*'\x1b[1A\x1b[2K')
    print( '\033[0;0H')
    print( '\b\033[4mHost:', host_name, '\033[0m')

    # Get CPU info
    c0 = getoutput( "cat /proc/stat | head -n 1 | awk '{print $2, $3, $4, $5}'")
    c0 = numpy.fromstring( c0, sep=' ')
    v0 = [float(c0[0]-last_cpu_poll[0]) / float(sum(c0)-sum(last_cpu_poll)),
        float(c0[1]+c0[2]-last_cpu_poll[1]-last_cpu_poll[2]) / float(sum(c0)-sum(last_cpu_poll))]
    last_cpu_poll = c0

    # Get new disk sector counts
    d = getoutput( "cat /proc/diskstats | grep sd | awk {'print $3,$6,$10'}")
    cu = numpy.array( [[int(j[1]),int(j[2])] for j in [i.split(' ') for i in d.split('\n')] if j[0] in disk_names])
    ds = (cu-last_disk_poll).astype(float)
    ds = [ds[i]/(1024*1024/disk_sector_sizes[i]) for i in range( len( ds))]
    last_disk_poll = cu

    # Get memory usage
    m0 = getoutput( "cat /proc/meminfo | head -n 6 | awk '{print $2}'")
    m0 = numpy.fromstring( m0, sep='\n').astype( float) / 1024 / 1024
    m0 = [m0[0]-m0[2], m0[3]+m0[4], m0[0], m0[5]]

    # Show me
    print( 'Utilization %')
    print( '\033[36m', 'CPUs  :', sbar( v0[0], v0[1], wid), '\033[36m%3.1f\033[0m / \033[91m%3.1f' % (100*v0[0], 100*v0[1]), '\033[0m')
    for i in range( len( D)):
        print( '\033[92m', 'GPU', D[i][0], ':', ubar( D[i][1]/100, wid), D[i][1], '\033[0m')

    print( '\nMemory')
    print( '\033[36m', 'RAM   :', dbar( m0[0]/m0[2], m0[3], wid), '%4.1fGiB\033[0m / \033[91m%3.1fGiB' % (m0[0], m0[3]), '\033[0m')
    for i in range( len( D)):
        print( '\033[93m', 'GPU', D[i][0], ':', dbar( D[i][2]/D[i][5], D[i][3]/100, wid), \
            '%4.1fGiB / %2d%%' % (D[i][2]/1024,D[i][3]), '\033[0m')

    print( '\nTemperature C')
    for i in range( len( D)):
        print( '\033[91m', 'GPU', D[i][0], ':', ubar( D[i][4]/100, wid), D[i][4], '°C \033[0m')

    print( '\nDisk I/O:')
    for i in range( len( disk_names)):
        print( '\033[36m', disk_names[i], ' : ', dbar( ds[i][0]/6000, ds[i][1]/6000, wid), \
            '\033[92m%4.1fMiB\033[0m / \033[91m%3.1fMiB' % (ds[i][0], ds[i][1]), '\033[0m')


# Draw the results
def drawh( D, wid=20):

    global last_cpu_poll
    global host_name
    global disk_names
    global disk_sector_sizes
    global last_disk_poll

    # Remove previous output
    #print (len(D)+5)*'\x1b[1A\x1b[2K'
    print( '\033[0;0H')
    print( '\b\033[4mHost:', host_name, '\033[0m')

    # Get CPU info
    c0 = getoutput( "cat /proc/stat | head -n 1 | awk '{print $2, $3, $4, $5}'")
    c0 = numpy.fromstring( c0, sep=' ')
    v0 = [float(c0[0]-last_cpu_poll[0]) / float(sum(c0)-sum(last_cpu_poll)),
        float(c0[1]+c0[2]-last_cpu_poll[1]-last_cpu_poll[2]) / float(sum(c0)-sum(last_cpu_poll))]
    last_cpu_poll = c0

    # Get new disk sector counts
    d = getoutput( "cat /proc/diskstats | grep sd | awk {'print $3,$6,$10'}")
    cu = numpy.array( [[int(j[1]),int(j[2])] for j in [i.split(' ') for i in d.split('\n')] if j[0] in disk_names])
    ds = (cu-last_disk_poll).astype(float)
    ds = [ds[i]/(1024*1024/disk_sector_sizes[i]) for i in range( len( ds))]
    last_disk_poll = cu

    # Get memory usage
    m0 = getoutput( "cat /proc/meminfo | head -n 6 | awk '{print $2}'")
    m0 = numpy.fromstring( m0, sep='\n').astype( float) / 1024 / 1024
    m0 = [m0[0]-m0[2], m0[3]+m0[4], m0[0], m0[5]]

    # Show me the GPUs, one per row
    print( 'Dev     \033[92mUtilization %\033[0m', ' '*(wid-7), '\033[93mMemory\033[0m', ' '*(wid+10), '\033[91mTemperature C\033[0m')
    for i in range( len( D)):
        print( 'GPU', D[i][0], ' \033[92m', ubar( float(99.*D[i][1]/100)/100, wid), '%3d'%(D[i][1]), '\033[0m', \
        '\033[93m', dbar( float(D[i][2])/D[i][5], float(D[i][3])/100, wid), \
            '%4.1fGiB / %2d%%' % (float( D[i][2])/1024,D[i][3]), '\033[0m', \
        '\033[91m', ubar( float(D[i][4])/100, wid), D[i][4], '°C \033[0m')

    # Show CPU and RAM
    print( '\033[36mCPU/RAM', sbar( v0[0], v0[1], wid), '\033[36m%3.0f\033[0m ' % (100*(v0[0]+v0[1])), '\033[0m\033[36m', \
        dbar( m0[0]/m0[2], m0[3], wid), '%4.1fGiB / %3.1fGiB' % (m0[0], m0[3]), '\033[0m')

    # Show disks
    dl = ''
    for i in range( len( disk_names)):
        dl += '\033[36m'+disk_names[i] + ('\033[92m%4.1fMiB\033[0m / \033[91m%3.1fMiB' % (ds[i][0], ds[i][1])) +  '\033[0m\t'
    print( dl)

# Draw compact results
def drawc( D, wid=20):

    global last_cpu_poll
    global host_name
    global disk_names
    global disk_sector_sizes
    global last_disk_poll

    # Remove previous output
    #print( 4*'\x1b[1A\x1b[2K')
    print( '\033[0;0H')

    # Get CPU info
    c0 = getoutput( "cat /proc/stat | head -n 1 | awk '{print $2, $3, $4, $5}'")
    c0 = numpy.fromstring( c0, sep=' ')
    v0 = [float(c0[0]-last_cpu_poll[0]) / float(sum(c0)-sum(last_cpu_poll)),
        float(c0[1]+c0[2]-last_cpu_poll[1]-last_cpu_poll[2]) / float(sum(c0)-sum(last_cpu_poll))]
    last_cpu_poll = c0

    # Get memory usage
    m0 = getoutput( "cat /proc/meminfo | head -n 6 | awk '{print $2}'")
    m0 = numpy.fromstring( m0, sep='\n').astype( float) / 1024 / 1024
    m0 = [m0[0]-m0[2], m0[3]+m0[4], m0[0], m0[5]]

    # If we have less than 4 GPUs add some zeros, I should do somethign for > 4 GPUs too!
    if len( D) < 3:
        D += [[0,0,0,0,0,0]]*(4-len(D))

    # Show me
    print( '\033[36mCPU/RAM ', qbar( [v0[0]+v0[1],m0[0]/m0[2]], wid), '\033[0m', \
        ' \b\033[4mHost:', host_name, '\033[0m')
    print( '\033[92mGPU Util', qbar( [float(D[0][1])/100,float(D[1][1])/100,float(D[2][1])/100,float(D[3][1])/100], wid), '\033[0m', \
        '\033[91mGPU Temp', qbar( [float(D[0][4])/100,float(D[1][4])/100,float(D[2][4])/100,float(D[3][4])/100], wid), '\033[0m')
    print( '\033[93mGPU Mem ', qbar( [float(D[0][1])/100,float(D[1][1])/100,float(D[2][1])/100,float(D[3][1])/100], wid), '\033[0m', \
        '\033[95mGPU R/W ', qbar( [float(D[0][3])/100,float(D[1][3])/100,float(D[2][3])/100,float(D[3][3])/100], wid), '\033[0m')

#
# Do it
#

from argparse import RawTextHelpFormatter

def main():
    parser = argparse.ArgumentParser( description="""GPU Usage Monitor

    Most bars are self explanatory and show %'s, with value on the right.

    \033[92mGPU 3 : ██████████████      | 70\033[0m

    Temperature is in Celcius.

    Memory reporting for the GPUs is shown using a double bar.
    Top bar is % of memory used, and the bottom bar is memory
    utilization as a % of the card capacity. Their values are
    also shown as numbers on the right of the bars.

      \033[93mGPU 0 : ███▀                |  2.8GiB / 15%\033[0m

    For the host CPU you will get a stacked blue and red bar. The blue
    represents the % of user time, and the red represents the % of the
    system time. Both values are shown on the right as numbers as well

      \033[36mCPUs  : ██▏                 | 14.9\033[0m / \033[91m1.2\033[0m

    For host RAM tou will get a double bar. The top half represents
    the used RAM % and the bottom one the amount of swap space.

      \033[36mRAM   : ▀                   |  5.4GiB\033[0m \033[91m/ 0.0GiB\033[0m

    In horizontal view, host CPU and RAM are shown in one line.
    CPU number shows user+system % usage, RAM shows used RAM and swap space:

      \033[36mCPU/RAM ██▎                 | 15   ▀                   |  5.4GiB / 0.0GiB\033[0m

    Compact mode requires a Braille unicode font (Macs have it). All
    bars will be shown in one line. E.g. for 4 GPUS you get:

      \033[91mGPU Temp |⣿⣿⣿⣿⣿⣿⣿⣿⠭⠭⠭⠭        |\033[0m

    In compact mode the host CPU/RAM will be shown as a double bar.

    """,
                                       usage='%(prog)s [OPTIONS]',
                                       formatter_class=RawTextHelpFormatter)

    parser.add_argument( '--vertical', '-V', action='store_true', default=False,
                        help='Use vertical format')
    parser.add_argument( '--horizontal', '-H', action='store_true', default=False,
                        help='Use horizontal format')
    parser.add_argument( '--compact', '-C', action='store_true', default=False,
                        help='Use compact format')
    parser.add_argument( '--width', '-w', type=int, default=10,
                        help='Width of bars in characters')
    args = parser.parse_args()

    # If undefined use horizontal orientation
    if args.horizontal is False and args.vertical is False and args.compact is False:
        args.horizontal = True

    global host_name
    global last_cpu_poll
    global disk_names
    global disk_sector_sizes
    global last_disk_poll

    # Get the hostname
    host_name = socket.gethostname().upper()

    # Init CPU usage counters
    d0 = getoutput( "cat /proc/stat | head -n 1 | awk '{print $2, $3, $4, $5}'")
    last_cpu_poll = numpy.fromstring( d0, sep=' ')

    # Get names of available disk devices and their sector sizes
    d = getoutput( "df | grep /dev/sd | awk {\'print $1\'}")
    disk_names = [i[5:-1] if i[-1].isdigit() else i[5:] for i in d.split('\n') if len(i) > 1]
    disk_names =  list( filter( None, disk_names))
    disk_sector_sizes = [int(getoutput( 'cat /sys/block/' + i + '/queue/hw_sector_size')) for i in disk_names]
    disk_names = list( set( disk_names))
    disk_sector_sizes = list( set( disk_sector_sizes))
    if len( disk_sector_sizes) == 1:
        disk_sector_sizes = [disk_sector_sizes[0] for _ in range( len( disk_names))]

    # Get initial sector counts
    d = getoutput( "cat /proc/diskstats | grep sd | awk {'print $3,$6,$10'}")
    last_disk_poll = numpy.array( [[int(j[1]),int(j[2])] for j in [i.split(' ') for i in d.split('\n')] if j[0] in disk_names])

    # Start nvidia-smi subprocess
    cmd = 'nvidia-smi -l 1 --query-gpu=count,index,utilization.gpu,memory.used,utilization.memory,temperature.gpu,memory.total --format=csv,noheader,nounits'
    master,slave = pty.openpty()
    Popen( cmd, shell=True, stdin=PIPE, stdout=slave, stderr=slave, close_fds=True)
    stdout = os.fdopen( master)

    # Clear the screen
    print( '\033[2J')

    # Start polling it
    D = []
    while True:
        # Get all the data in place
        t = list( map( int, stdout.readline().split(',')))

        # If this is the first call allocate GPU data array and do a buffer draw
        if len( D) == 0:
            D = [None]*t[0]
            if args.horizontal:
                print( (len(D)+4)*'\n')
            elif args.compact:
                print( 3*'\n')
            elif args.vertical:
                print( (3*(len(D)+2)+2)*'\n')
        D[t[1]] = t[1:]

        # Time to report
        if t[1] == t[0]-1:
            if args.horizontal:
                drawh( D, args.width)
            elif args.compact:
                drawc( D, args.width)
            elif args.vertical:
                draw( D, args.width)

# Do it
if __name__ == "__main__":
    main()
