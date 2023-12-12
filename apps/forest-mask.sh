# TODO script currently only works when in Eolab folder, but this placement doesn't make sense.
# Should be incorporated into NF workflow anyway
set -e

docker run --rm -v $PWD:/lol ghcr.io/osgeo/gdal:alpine-normal-latest \
  sh -c "ogrinfo -json -ro /lol/gadm41_DEU.gpkg ADM_ADM_0 > /lol/info.json"

BBOX=$(cat info.json | jq '.layers[] | .geometryFields[] | .extent[]')
LOWER_LONGITUDE=$(bc <<< "x=$(echo $BBOX | cut -d ' ' -f1)/1;x-(x%3)")
UPPER_LONGITUDE=$(bc <<< "x=$(echo $BBOX | cut -d ' ' -f3)/1;x-(x%3)")
LOWER_LATITUDE=$(bc  <<< "x=$(echo $BBOX | cut -d ' ' -f2)/1;x-(x%3)")
UPPER_LATITUDE=$(bc  <<< "x=$(echo $BBOX | cut -d ' ' -f4)/1;x-(x%3)")

for ((i=LOWER_LATITUDE; i<=UPPER_LATITUDE; i+=3)); do
  for ((j=LOWER_LONGITUDE; j<=UPPER_LONGITUDE; j+=3)); do
    #! THIS ONLY WORKS FOR THE NOTHERN HEMISPHERE, EAST OF GREENWICH
    printf "ESA_WorldCover_10m_2020_v100_N%.2dE%.3d_Map.tif\n" $i $j >> ./world-cover.txt
  done
done

if ! [ -d masks ]; then
  echo "Output directory needs to exist"
  exit 1
fi

while read -r line; do
  docker run -u "$(id -u):$(id -g)" --env LINE=${line} --rm -v $PWD:/data/ -v /codede/auxdata/esa-worldcover-2020:/worldcover/ davidfrantz/force \
    bash -c 'force-cube -s 10 -n 0 -o /data/masks/ -b "worldcover" -j 8 "/worldcover/${LINE}"'
done < ./world-cover.txt

# TODO Don't hardcode blocksizes
find Eolab/masks -name '*tif' -type f -execdir \
  gdal_calc.py --outfile=mask.tif --calc="1*logical_and(A>=10,A<20)" \
  --type=Byte --NoDataValue=0 --creation-option="BLOCKXSIZE=300" --creation-option="BLOCKYSIZE=3000" \
  --overwrite -A {} +
