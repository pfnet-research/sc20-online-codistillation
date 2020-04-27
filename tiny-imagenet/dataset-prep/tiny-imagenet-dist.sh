set -eux

BASEDIR=~
SRCDIR=${BASEDIR}/tiny-imagenet-200
DESTDIR=${BASEDIR}/tiny-imagenet-200-converted

rm -rf $SRCDIR
unzip -q ${BASEDIR}/tiny-imagenet-200.zip -d $BASEDIR
rm -rf $DESTDIR
mkdir -p $DESTDIR

echo "Transferring train data"
mkdir ${DESTDIR}/train

for CLS in `ls ${SRCDIR}/train`
do
    cp -r ${SRCDIR}/train/${CLS}/images/ ${DESTDIR}/train/${CLS}
done


echo "Transferring val data"

DIST_VAL_SH="/tmp/_tiny-imagenet-dist-val.sh"
rm -f $DIST_VAL_SH
mkdir ${DESTDIR}/val
python tiny-imagenet-dist-val.py ${SRCDIR}/val ${DESTDIR}/val > $DIST_VAL_SH
chmod +x $DIST_VAL_SH
${DIST_VAL_SH}

echo "Done!"
