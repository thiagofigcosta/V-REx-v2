#   Copyright 2012 Kapil Thangavelu
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# Pytho{\}: Ignore file

import pymongo
import contextlib
import time
import uuid

from pymongo import errors

from datetime import datetime, timedelta
import traceback


DEFAULT_INSERT = {
    "attempts": 0,
    "locked_by": None,
    "locked_at": None,
    "last_error": None
}


class MongoQueue(object):
    """A queue class
    """

    def __init__(self, collection, consumer_id, timeout=300, max_attempts=3):
        """
        """
        self.collection = collection
        self.consumer_id = consumer_id
        self.timeout = timeout
        self.max_attempts = max_attempts

    def close(self):
        """Close the in memory queue connection.
        """
        self.collection.connection.close()

    def clear(self):
        """Clear the queue.
        """
        return self.collection.drop()

    def size(self):
        """Total size of the queue
        """
        return self.collection.count()

    def repair(self):
        """Clear out stale locks.

        Increments per job attempt counter.
        """
        self.collection.find_and_modify(
            query={
                "locked_by": {"$ne": None},
                "locked_at": {
                    "$lt": datetime.now() - timedelta(self.timeout)}},
            update={
                "$set": {"locked_by": None, "locked_at": None},
                "$inc": {"attempts": 1}}
        )

    def drop_max_attempts(self):
        """
        """
        self.collection.find_and_modify(
            {"attempts": {"$gte": self.max_attempts}},
            remove=True)

    def put(self, payload, priority=0):
        """Place a job into the queue
        """
        job = dict(DEFAULT_INSERT)
        job['priority'] = priority
        job['payload'] = payload
        return self.collection.insert(job)

    def next(self):
        return self._wrap_one(self.collection.find_and_modify(
            query={"locked_by": None,
                   "locked_at": None,
                   "attempts": {"$lt": self.max_attempts}},
            update={"$set": {"locked_by": self.consumer_id,
                             "locked_at": datetime.now()}},
            sort=[('priority', pymongo.DESCENDING)],
            new=1,
            limit=1
        ))

    def _jobs(self):
        return self.collection.find(
            query={"locked_by": None,
                   "locked_at": None,
                   "attempts": {"$lt": self.max_attempts}},
            sort=[('priority', pymongo.DESCENDING)],
        )

    def _wrap_one(self, data):
        return data and MongoJob(self, data) or None

    def stats(self):
        """Get statistics on the queue.

        Use sparingly requires a collection lock.
        """

        js = """function queue_stat(){
        return db.eval(
        function(){
           var a = db.%(collection)s.count(
               {'locked_by': null,
                'attempts': {$lt: %(max_attempts)i}});
           var l = db.%(collection)s.count({'locked_by': /.*/});
           var e = db.%(collection)s.count(
               {'attempts': {$gte: %(max_attempts)i}});
           var t = db.%(collection)s.count();
           return [a, l, e, t];
           })}""" % {
             "collection": self.collection.name,
             "max_attempts": self.max_attempts}

        return dict(zip(
            ["available", "locked", "errors", "total"],
            self.collection.database.eval(js)))


class MongoJob(object):

    def __init__(self, queue, data):
        """
        """
        self._queue = queue
        self._data = data

    @property
    def payload(self):
        return self._data['payload']

    @property
    def job_id(self):
        return self._data['_id']

    @property
    def priority(self):
        return self._data["priority"]

    @property
    def attempts(self):
        return self._data["attempts"]

    @property
    def locked_by(self):
        return self._data["locked_by"]

    @property
    def locked_at(self):
        return self._data["locked_at"]

    @property
    def last_error(self):
        return self._data["last_error"]

    ## MongoJob Control

    def complete(self):
        """MongoJob has been completed.
        """
        return self._queue.collection.find_and_modify(
            {'_id': self.job_id, "locked_by": self._queue.consumer_id},
            remove=True)

    def error(self, message=None):
        """Note an error processing a job, and return it to the queue.
        """
        self._queue.collection.find_and_modify(
            {'_id': self.job_id, "locked_by": self._queue.consumer_id},
            update={"$set": {
                "locked_by": None, "locked_at": None, "last_error": message},
                "$inc": {"attempts": 1}})

    def progress(self, count=0):
        """Note progress on a long running task.
        """
        return self._queue.collection.find_and_modify(
            {'_id': self.job_id, "locked_by": self._queue.consumer_id},
            update={"$set": {"progress": count, "locked_at": datetime.now()}})

    def release(self):
        """Put the job back into_queue.
        """
        return self._queue.collection.find_and_modify(
            {'_id': self.job_id, "locked_by": self._queue.consumer_id},
            update={"$set": {"locked_by": None, "locked_at": None},
                    "$inc": {"attempts": 1}})
    
    def put_back(self):
        """Put the job back into_queue without attempt.
        """
        return self._queue.collection.find_and_modify(
            {'_id': self.job_id, "locked_by": self._queue.consumer_id},
            update={"$set": {"locked_by": None, "locked_at": None,"priority":-1}})

    ## Context Manager support

    def __enter__(self):
        return self._data

    def __exit__(self, type, value, tb):
        if (type, value, tb) == (None, None, None):
            self.complete()
        else:
            error = traceback.format_exc()
            self.error(error)


@contextlib.contextmanager
def lock(collection, key, wait=30, poll_period=5, lease_period=30):

    lock = MongoLock(collection, key, lease_period)
    try:
        lock.acquire(wait=wait, poll_period=poll_period)
        yield lock
    finally:
        lock.release()


class MongoLock(object):

    def __init__(self, collection, lock_name, lease=120):
        self.collection = collection
        self.lock_name = lock_name
        self._client_id = uuid.uuid4().hex
        self._locked = False
        self._lease_time = lease
        self._lock_expires = False

    @property
    def locked(self):
        if not self._locked:
            return self._locked
        valid=datetime.now() < self._lock_expires
        if not valid:
            self.release()
        return self._locked and valid

    @property
    def client_id(self):
        return self._client_id

    def fetch(self):
        result=self.collection.find({'_id': self.lock_name})
        if result.count()>0:
            lock_on_db=result.next()
            self._lock_expires=lock_on_db['ttl']
            self._client_id=lock_on_db['client_id']
            self._locked = True
        else:
            self._lock_expires=False
            self._locked = False

    def refresh(self):
        ttl = datetime.now() + timedelta(seconds=self._lease_time)
        try:
            self.collection.update_one({'_id': self.lock_name, 'client_id': self._client_id}, { '$set': { 'ttl': ttl }})
            self._lock_expires = ttl
            self._locked = True
        except Exception:
            pass
        return self._locked

    def acquire(self, wait=None, poll_period=5):
        result = self._acquire()
        if not wait:
            return result

        assert isinstance(wait, int)
        max_wait = datetime.now() + timedelta(wait)
        while max_wait < datetime.now():
            result = self._acquire()
            if result:
                return result
            time.sleep(poll_period)

    def _acquire(self):
        ttl = datetime.now() + timedelta(seconds=self._lease_time)
        try:
            self.collection.insert({
                '_id': self.lock_name,
                'ttl': ttl,
                'client_id': self._client_id},
                w=1, j=True)
        except errors.DuplicateKeyError:
            self.collection.remove(
                {'_id': self.lock_name, 'ttl': {'$lt': datetime.now()}})
            try:
                self.collection.insert(
                    {'_id': self.lock_name,
                     'ttl': ttl,
                     'client_id': self._client_id}, w=1, j=True)
            except errors.DuplicateKeyError:
                self._locked = False
                return self._locked
        self._lock_expires = ttl
        self._locked = True
        return self._locked

    def release(self):
        if not self._locked:
            return False
        self.collection.remove(
            {'_id': self.lock_name, 'client_id': self._client_id}, j=True, w=1)
        self._locked = False
        return True
