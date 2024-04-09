echo "converting .mid to .ly..."
/home/evanmm3/bin/midi2ly -d $2 -o outputs/"$1"/"$1$2".ly "$1".mid

echo "converting .ly to .pdf..."
/home/evanmm3/bin/lilypond -o outputs/"$1"/"$1$2" outputs/"$1"/"$1$2".ly