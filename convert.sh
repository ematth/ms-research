echo "converting .mid to .ly..."

/home/evanmm3/bin/midi2ly -d $2 -o outputs/"$1$2".ly "$1".mid # -t 1*1/1 

echo "converting .ly to .pdf..."
/home/evanmm3/bin/lilypond -s -fpng -dcrop="#t" -o outputs/"$1$2" outputs/"$1$2".ly #-dpng-width="595" -dpng-height="842"