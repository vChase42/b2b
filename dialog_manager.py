from datetime import datetime
from threading import Lock

class DialogManager:
    def __init__(self):
        self.blurbs = []  # List to store text blurbs
        self.lock = Lock()  # Thread safety lock
    
        self.latest_blurb = {}

    def add_blurb(self, text, speaker_name=None, start_time=None, end_time=None):
        if start_time is None:
            start_time = datetime.now()
        
        blurb = {
            'text': text,
            'speaker_name': speaker_name,
            'start_time': start_time,
            'end_time': end_time
        }

        with self.lock:
            self.blurbs.append(blurb)
        
    def set_curr_blurb(self, text, speaker_name=None, start_time=None, end_time=None):
        blurb = {
            'text': text,
            'speaker_name': speaker_name,
            'start_time': start_time,
            'end_time': end_time
        }
        self.latest_blurb = blurb


    def edit_blurb(self, index, text=None, speaker_name=None, start_time=None, end_time=None):
        with self.lock:
            if text is not None:
                self.blurbs[index]['text'] = text
            if speaker_name is not None:
                self.blurbs[index]['speaker_name'] = speaker_name
            if start_time is not None:
                self.blurbs[index]['start_time'] = start_time
            if end_time is not None:
                self.blurbs[index]['end_time'] = end_time

    def get_blurb(self, index):
        return self.blurbs[index]

    def to_string(self):
        output = []
        for blurb in self.blurbs:
            start_str = blurb['start_time'].strftime("[%H:%M:%S]")
            end_str = blurb['end_time'].strftime("[%H:%M:%S]") if blurb['end_time'] else ""
            speaker_str = f" {blurb['speaker_name']}" if blurb['speaker_name'] else ""
            if end_str:
                output.append(f"{start_str}-{end_str}{speaker_str}: {blurb['text']}")
            else:
                output.append(f"{start_str}{speaker_str}: {blurb['text']}")
        return "\n".join(output)

    def find_by_time(self, time):
        for index, blurb in enumerate(self.blurbs):
            if blurb['start_time'] <= time and (blurb['end_time'] is None or blurb['end_time'] >= time):
                return index, blurb
        return None, None

    def edit_by_time(self, time, text=None, speaker_name=None, start_time=None, end_time=None):
        index, blurb = self.find_by_time(time)
        if blurb:
            self.edit_blurb(index, text=text, speaker_name=speaker_name, start_time=start_time, end_time=end_time)
        else:
            print(f"No blurb found at time {time}")


    def clear(self):
        self.blurbs = []

# Example Usage
if __name__ == "__main__":
    manager = DialogManager()
    start = datetime(2024, 10, 2, 14, 0, 0)
    end = datetime(2024, 10, 2, 14, 1, 30)
    
    manager.add_blurb("Hello, world!", "Speaker1", start, end)
    manager.add_blurb("Goodbye!", "Speaker2", start)

    print(manager.to_string())

    # Finding a blurb at a specific time
    search_time = datetime(2024, 10, 2, 14, 0, 30)
    index, blurb = manager.find_by_time(search_time)
    if blurb:
        print(f"Found blurb: {blurb}")
    else:
        print(f"No blurb found at {search_time}")

    # Editing a blurb at a specific time
    manager.edit_by_time(search_time, text="Hello again!", speaker_name="Speaker1", end_time=end)
    print(manager.to_string())
