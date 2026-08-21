package com.v10.intenttester;

import android.app.Activity;
import android.os.Bundle;
import android.text.InputType;
import android.graphics.Color;
import android.view.*;
import android.widget.*;
import org.tensorflow.lite.Interpreter;
import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;
import org.json.*;

public class MainActivity extends Activity {
    Interpreter interpreter; Map<String,Integer> vocab; JSONArray labels; int maxLen=10; float threshold=0.6f;
    EditText input; TextView result, confidence;

    @Override public void onCreate(Bundle b){ super.onCreate(b); buildUI(); try{loadModel();}catch(Exception e){result.setText("Model load error: "+e.getMessage());} }

    void buildUI(){
        LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setPadding(28,40,28,28); root.setBackgroundColor(Color.rgb(248,250,252));
        TextView title=new TextView(this); title.setText("V10 Intent Tester"); title.setTextSize(28); title.setTextColor(Color.rgb(17,24,39)); title.setTypeface(null,1);
        TextView sub=new TextView(this); sub.setText("Offline • V10 TFLite model"); sub.setTextSize(15); sub.setTextColor(Color.DKGRAY); sub.setPadding(0,6,0,30);
        input=new EditText(this); input.setHint("Type a message…"); input.setTextSize(18); input.setSingleLine(false); input.setMinLines(2); input.setInputType(InputType.TYPE_CLASS_TEXT|InputType.TYPE_TEXT_FLAG_CAP_SENTENCES); 
        Button test=new Button(this); test.setText("TEST INTENT"); test.setOnClickListener(v->predict(input.getText().toString()));
        result=new TextView(this); result.setText("Intent: —"); result.setTextSize(24); result.setTypeface(null,1); result.setPadding(0,35,0,8);
        confidence=new TextView(this); confidence.setText("Confidence: —"); confidence.setTextSize(18); confidence.setTextColor(Color.DKGRAY);
        TextView examples=new TextView(this); examples.setText("\nTry:\nhello\nkaise ho\nmujhe madad chahiye\nshukriyaa\nxyzabc"); examples.setTextSize(16); examples.setTextColor(Color.DKGRAY);
        root.addView(title);root.addView(sub);root.addView(input,new LinearLayout.LayoutParams(-1,-2));root.addView(test);root.addView(result);root.addView(confidence);root.addView(examples); setContentView(root);
    }

    void loadModel() throws Exception {
        interpreter=new Interpreter(loadAsset("v10_intent_model.tflite"));
        String v=readAsset("v10_vocabulary.json"); JSONObject o=new JSONObject(v); vocab=new HashMap<>(); Iterator<String> it=o.keys(); while(it.hasNext()){String k=it.next();vocab.put(k,o.getInt(k));}
        labels=new JSONArray(readAsset("v10_labels.json")); JSONObject c=new JSONObject(readAsset("v10_config.json")); maxLen=c.optInt("max_len",10); threshold=(float)c.optDouble("unknown_threshold",0.6);
    }
    ByteBuffer loadAsset(String name)throws Exception{ AssetFileDescriptorLike a=new AssetFileDescriptorLike(getAssets().openFd(name)); FileInputStream in=a.in; FileChannel ch=in.getChannel(); ByteBuffer buf=ch.map(FileChannel.MapMode.READ_ONLY,a.start,a.length); return buf; }
    String readAsset(String n)throws Exception{ InputStream in=getAssets().open(n); ByteArrayOutputStream out=new ByteArrayOutputStream(); byte[] b=new byte[4096]; int x; while((x=in.read(b))!=-1)out.write(b,0,x); return out.toString("UTF-8"); }

    void predict(String text){
        if(text==null||text.trim().isEmpty()){result.setText("Intent: —");confidence.setText("Confidence: —");return;}
        int[] ids=new int[maxLen]; String[] words=text.toLowerCase(Locale.ROOT).trim().split("\\s+"); int p=0; for(String w:words){ if(p>=maxLen)break; Integer id=vocab.get(w); ids[p++]=id==null?1:id; }
        float[][] out=new float[1][labels.length()]; interpreter.run(new int[][]{ids},out); int best=0; for(int i=1;i<out[0].length;i++)if(out[0][i]>out[0][best])best=i;
        float conf=out[0][best]; String label; try{label=labels.getString(best);}catch(Exception e){label="Unknown";}
        if(conf < threshold) label="Unknown";
        result.setText("Intent: "+label); confidence.setText(String.format(Locale.US,"Confidence: %.1f%%",conf*100f));
    }
    @Override protected void onDestroy(){if(interpreter!=null)interpreter.close();super.onDestroy();}
    static class AssetFileDescriptorLike{FileInputStream in;long start,length;AssetFileDescriptorLike(android.content.res.AssetFileDescriptor a){in=new FileInputStream(a.getFileDescriptor());start=a.getStartOffset();length=a.getLength();}}
}
