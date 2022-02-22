#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

/Users/spartee/.virtualenvs/smartsim/bin/python -m smartsim._core.entrypoints.colocated +ifname lo +lockfile smartsim-a308962.lock +db_cpus 1 +command /Users/spartee/Dropbox/Cray/smartsim/smartsim/_core/bin/redis-server /Users/spartee/Dropbox/Cray/smartsim/smartsim/_core/config/redis6.conf --loadmodule /Users/spartee/Dropbox/Cray/smartsim/smartsim/_core/lib/redisai.so --port 6780 --logfile /dev/null &
DBPID=$!

$@

