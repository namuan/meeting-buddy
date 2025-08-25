# Meeting Buddy

### Prerequisites

- Python 3.9 or higher

### Pre-Requisites

#### BlackHole

```
brew install blackhole-2ch
```

This may be required if you are not able to see "BlackHold 2ch" in "Audio MIDI Setup" application.

```
sudo killall coreaudiod
```

#### Audio MIDI Setup

Open "/System/Applications/Utilities/Audio MIDI Setup.app"

Add a new Device and select "BlackHole 2ch" and the other device where you'd like to route the output.

![](assets/audio-midi-setup.png)

#### Use Aggregated Device for Output

Open "Sounds" preferences and select the new aggregated device as output.

![](assets/sound-output-device.png)

### Installation

```bash
# Clone the repository
git clone https://github.com/namuan/meeting-buddy.git
cd meeting-buddy

# Install dependencies using uv
make install
```

### Running the Application

```bash
# Run the application directly
make run

# Or run using uv
uv run meeting-buddy
```

## License

This project is licensed under the [MIT License](LICENSE).
