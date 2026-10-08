Put the model here before building:

  model.pte    the CPU / XNNPACK program (NOT model_qnn.pte): from this repository's
               GitHub release, your deploy.zip, or the learning block's output
  labels.txt   from the same place, replacing the placeholder in this folder

labels.txt must be in the model's OUTPUT order, one class per line. The placeholder
(Paper, Rock, Scissors, unknown) is Edge Impulse's alphabetical order for the
rock-paper-scissors project; check it against your own model's labels.

Optional: selftest.jpg - a photo of a known gesture, used by the "Self-test" button to
compare confidences with Python (see the README, "Parity check").

You can also skip rebuilding and swap files at runtime with adb:
  adb push model.pte labels.txt /sdcard/Android/data/com.demo.rpspte/files/
