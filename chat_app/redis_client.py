import redis

class RedisClient:
    def __init__(self):
        self.client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    def get_session_count(self, username):
        user_session_count_key = f"{username}_session_count"
        count = self.client.get(user_session_count_key)
        if not count:
            count = 0
        return count
    
    def increment_user_session_count(self, username):
        user_session_count_key = f"{username}_session_count"
        count = self.client.get(user_session_count_key)
        if count:
            count = int(count) + 1
        else:
            count = 1
        self.client.set(user_session_count_key, count)
        return count
    
    def get_session_details(self, session_id):
        session_details = self.client.lrange(session_id, 0, -1)
        return session_details
    
    def get_active_user_sessions(self, username):
        keys = self.client.keys(f'message_store:{username}:*')
        return keys
    
    def set_alias(self, alias):
        self.client.mset(alias)
    
    def get_alias(self, key):
        return self.client.get(key)
    
    def delete(self, key):
        return self.client.delete(key)