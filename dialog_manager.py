from datetime import datetime
from threading import Lock

class DialogManager:
    def __init__(self):
        self.blurbs = []  # List to store text blurbs
        self.lock = Lock()  # Thread safety lock
    
    def sort_by_time(self):
        with self.lock:
            self.blurbs.sort(key=lambda blurb: blurb['start_time'])



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
        self.sort_by_time()
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
            latest_blurb_or_within_end = (blurb['end_time'] is None or blurb['end_time'] >= time)
            if blurb['start_time'] <= time and latest_blurb_or_within_end: 
                return index
                
        return None

    def edit_by_time(self, time, text=None, speaker_name=None, start_time=None, end_time=None):
        index = self.find_by_time(time)
        if index is None:
            print(f"No blurb found at time {time}")
            return

        self.edit_blurb(index, text=text, speaker_name=speaker_name, start_time=start_time, end_time=end_time)

    def get_text_before_time(self,time):
        index = self.find_by_time(time)
        if index == None:
            return ""
        text = ""
        for i in range(index):
            text+= self.blurbs[i]['text'] + " "
        return text

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
