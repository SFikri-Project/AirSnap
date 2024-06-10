package com.example.imgclassifierapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.example.imgclassifierapp.ml.ModelUnquant; // Ensure this is correct
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence, recommendation;
    ImageView imageView;
    Button picture, gallery;
    int imageSize = 224;
    AlertDialog guideDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        recommendation = findViewById(R.id.recommendation);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        showCameraGuideDialog();

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dismissGuideDialog();
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dismissGuideDialog();
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 2);
            }
        });
    }
    private void showCameraGuideDialog() {
        // Create dialog
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        // Set layout for dialog
        View view = getLayoutInflater().inflate(R.layout.dialog_camera_guide, null);
        builder.setView(view);
        // Show dialog
        guideDialog = builder.create();
        guideDialog.show();
    }

    // Method to dismiss guide dialog
    private void dismissGuideDialog() {
        if (guideDialog != null && guideDialog.isShowing()) {
            guideDialog.dismiss();
        }
    }

    public void classifyImage(Bitmap image) {
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext()); // Ensure the model class name is correct

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Baik", "Sedang", "Tidak Sehat untuk Kelompok Tertentu", "Tidak Sehat", "Sangat Tidak Sehat", "Parah"};
            result.setText(classes[maxPos]);
            showQualityExplanation(classes[maxPos]);


            // Set recommendations based on the classification
            String[] recommendations = {
                    "Kualitas udara baik. Anda bisa beraktivitas di luar ruangan dengan nyaman.",
                    "Kualitas udara sedang. Disarankan untuk tidak beraktivitas di luar ruangan terlalu lama.",
                    "Kualitas udara tidak sehat untuk kelompok tertentu. Orang yang sensitif harus mengurangi aktivitas di luar ruangan.",
                    "Kualitas udara tidak sehat. Semua orang harus membatasi aktivitas di luar ruangan.",
                    "Kualitas udara sangat tidak sehat. Disarankan untuk tinggal di dalam ruangan dan menghindari aktivitas fisik.",
                    "Kualitas udara parah. Semua orang harus menghindari aktivitas di luar ruangan dan tetap di dalam ruangan."
            };
            recommendation.setText(recommendations[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void showQualityExplanation(String qualityLabel) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Penjelasan Kualitas Udara");
        String explanation = "";

        switch (qualityLabel) {
            case "Baik":
                explanation = "Kualitas udara dianggap memuaskan, dan polusi udara menimbulkan sedikit atau tidak ada risiko. Nilai estimasi AQI berada pada rentang 0-50.";
                break;
            case "Sedang":
                explanation = "Kualitas udara sedang, dan beberapa orang yang sangat peka terhadap polusi udara mungkin akan mengalami efek kesehatan jika terpapar dalam jangka waktu yang lama. Nilai estimasi AQI berada pada rentang 51-100.";
                break;
            case "Tidak Sehat untuk Kelompok Tertentu":
                explanation = "Kualitas udara tidak sehat untuk kelompok tertentu, seperti orang dengan penyakit jantung atau penyakit paru-paru, anak-anak, dan orang tua. Nilai estimasi AQI berada pada rentang 101-150.";
                break;
            case "Tidak Sehat":
                explanation = "Kualitas udara tidak sehat, dan semua orang mungkin mulai merasakan efek kesehatan jika terpapar dalam jangka waktu yang lama. Nilai estimasi AQI berada pada rentang 151-200.";
                break;
            case "Sangat Tidak Sehat":
                explanation = "Kualitas udara sangat tidak sehat, dan kemungkinan besar semua orang akan merasakan efek kesehatan yang lebih serius. Nilai estimasi AQI berada pada rentang 201-300.";
                break;
            case "Parah":
                explanation = "Kualitas udara parah, dan efek kesehatan yang serius pada semua orang. Nilai estimasi AQI berada pada rentang 301-500.";
                break;
        }

        builder.setMessage(explanation);
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialogInterface, int i) {
                dialogInterface.dismiss();
            }
        });

        AlertDialog explanationDialog = builder.create();
        explanationDialog.show();
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            Bitmap image = null;
            if (requestCode == 1) {
                image = (Bitmap) data.getExtras().get("data");
            } else if (requestCode == 2 && data != null) {
                Uri uri = data.getData();
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (image != null) {
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
    }
}
