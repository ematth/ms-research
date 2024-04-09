echo "converting .mid to .ly..."
/home/evanmm3/bin/midi2ly -d $2 -o "$1"/"$1$2".ly "$1".mid

echo "converting .ly to .pdf..."
/home/evanmm3/bin/lilypond -o "$1"/"$1$2" "$1"/"$1$2".ly