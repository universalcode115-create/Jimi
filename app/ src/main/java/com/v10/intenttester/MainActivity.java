package com.v10.intenttester;

import android.app.Activity;
import android.os.Bundle;
import android.text.InputType;
import android.graphics.Color;
import android.view.*;
import android.widget.*;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;
import org.json.*;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;

public class MainActivity extends Activity {

    private Interpreter interpreter;
    private Map<String, Integer> vocab;
    private JSONArray labels;

    private int maxLen = 10;
    private float threshold = 0.6f;

    private EditText input;
    private TextView result;
    private TextView confidence;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        buildUI();

        try {
            loadModel();
            result.setText("Intent: Ready");
        } catch (Exception e) {
            result.setText("Model load error");
            confidence.setText(e.toString());
        }
    }

    private void buildUI() {

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(28, 40, 28, 28);
        root.setBackgroundColor(Color.rgb(248, 250, 252));

        TextView title = new TextView(this);
        title.setText("V10 Intent Tester");
        title.setTextSize(28);
        title.setTextColor(Color.rgb(17, 24, 39));
        title.setTypeface(null, 1);

        TextView sub = new TextView(this);
        sub.setText("Offline • V10 TFLite model");
        sub.setTextSize(15);
        sub.setTextColor(Color.DKGRAY);
        sub.setPadding(0, 6, 0, 30);

        input = new EditText(this);
        input.setHint("Type a message…");
        input.setTextSize(18);
        input.setMinLines(2);
        input.setInputType(
                InputType.TYPE_CLASS_TEXT |
                InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        );

        Button test = new Button(this);
        test.setText("TEST INTENT");
        test.setOnClickListener(v ->
                predict(input.getText().toString())
        );

        result = new TextView(this);
        result.setText("Intent: —");
        result.setTextSize(24);
        result.setTypeface(null, 1);
        result.setPadding(0, 35, 0, 8);

        confidence = new TextView(this);
        confidence.setText("Confidence: —");
        confidence.setTextSize(18);
        confidence.setTextColor(Color.DKGRAY);

        TextView examples = new TextView(this);
        examples.setText(
                "\nTry:\n" +
                "hello\n" +
                "kaise ho\n" +
                "mujhe madad chahiye\n" +
                "shukriyaa\n" +
                "xyzabc"
        );
        examples.setTextSize(16);
        examples.setTextColor(Color.DKGRAY);

        root.addView(title);
        root.addView(sub);
        root.addView(input,
                new LinearLayout.LayoutParams(-1, -2));
        root.addView(test);
        root.addView(result);
        root.addView(confidence);
        root.addView(examples);

        setContentView(root);
    }

    private void loadModel() throws Exception {

        interpreter = new Interpreter(loadModelFile());

        String vocabularyJson = readAsset("v10_vocabulary.json");

        JSONObject vocabularyObject =
                new JSONObject(vocabularyJson);

        vocab = new HashMap<>();

        Iterator<String> keys =
                vocabularyObject.keys();

        while (keys.hasNext()) {
            String key = keys.next();
            vocab.put(
                    key,
                    vocabularyObject.getInt(key)
            );
        }

        labels =
                new JSONArray(
                        readAsset("v10_labels.json")
                );

        JSONObject config =
                new JSONObject(
                        readAsset("v10_config.json")
                );

        maxLen =
                config.optInt("max_len", 10);

        threshold =
                (float) config.optDouble(
                        "unknown_threshold",
                        0.6
                );
    }

    private ByteBuffer loadModelFile() throws Exception {

        AssetFileDescriptor fileDescriptor =
                getAssets().openFd(
                        "v10_intent_model.tflite"
                );

        FileInputStream inputStream =
                new FileInputStream(
                        fileDescriptor.getFileDescriptor()
                );

        FileChannel fileChannel =
                inputStream.getChannel();

        long startOffset =
                fileDescriptor.getStartOffset();

        long declaredLength =
                fileDescriptor.getDeclaredLength();

        return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
        );
    }

    private String readAsset(String name)
            throws Exception {

        InputStream input =
                getAssets().open(name);

        ByteArrayOutputStream output =
                new ByteArrayOutputStream();

        byte[] buffer = new byte[4096];

        int length;

        while ((length = input.read(buffer)) != -1) {
            output.write(buffer, 0, length);
        }

        input.close();

        return output.toString("UTF-8");
    }

    private void predict(String text) {

        if (text == null ||
                text.trim().isEmpty()) {

            result.setText("Intent: —");
            confidence.setText("Confidence: —");
            return;
        }

        int[] ids = new int[maxLen];

        String[] words =
                text.toLowerCase(Locale.ROOT)
                        .trim()
                        .split("\\s+");

        int position = 0;

        for (String word : words) {

            if (position >= maxLen)
                break;

            Integer id = vocab.get(word);

            if (id == null) {
                id = 1;
            }

            ids[position] = id;

            position++;
        }

        float[][] output =
                new float[1][labels.length()];

        interpreter.run(
                new int[][]{ids},
                output
        );

        int best = 0;

        for (int i = 1;
             i < output[0].length;
             i++) {

            if (output[0][i] >
                    output[0][best]) {

                best = i;
            }
        }

        float conf =
                output[0][best];

        String label;

        try {
            label =
                    labels.getString(best);
        } catch (Exception e) {
            label = "Unknown";
        }

        if (conf < threshold) {
            label = "Unknown";
        }

        result.setText(
                "Intent: " + label
        );

        confidence.setText(
                String.format(
                        Locale.US,
                        "Confidence: %.1f%%",
                        conf * 100f
                )
        );
    }

    @Override
    protected void onDestroy() {

        if (interpreter != null) {
            interpreter.close();
        }

        super.onDestroy();
    }
             }
