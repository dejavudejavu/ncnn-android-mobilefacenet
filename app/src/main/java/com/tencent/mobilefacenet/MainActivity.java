// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.mobilefacenet;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;

public class MainActivity extends Activity
{
    private static final int SELECT_IMAGE = 1;
    private static final int SELECT_IMAGE_2 = 2;

    private TextView infoResult;
    private TextView timeResult;
    private ImageView imageView_1;
    private ImageView imageView_2;
    private Bitmap yourSelectedImage = null;
    private Bitmap yourSelectedImage_2 = null;


    private API api = new API();

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        boolean ret_init = api.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "mobilefacenet Init failed");
        }

        infoResult = (TextView) findViewById(R.id.textView2);
        timeResult=(TextView) findViewById(R.id.infoResult2);
        imageView_1 = (ImageView) findViewById(R.id.imageView);
        imageView_2=(ImageView) findViewById(R.id.imageView2);

        Button buttonImage1 = (Button) findViewById(R.id.buttonImg1);
        Button buttonImage2 = (Button) findViewById(R.id.buttonImg2);
        buttonImage1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });
        buttonImage2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE_2);
            }
        });

        Button buttonDetect = (Button) findViewById(R.id.buttonCompare);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;

                String result = api.Detect(yourSelectedImage,yourSelectedImage_2, false);
//                Log.i("result",result);

                if (result == null)
                {
                    infoResult.setText("detect failed");
                }
                else
                {
                    String[] strs=result.split("-");
                    infoResult.setText(strs[0]);
                    timeResult.setText(strs[1]);

                }
            }
        });
        Button buttonDetectgpu = (Button) findViewById(R.id.buttonDetectGPU);
        buttonDetectgpu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;

                String result = api.Detect(yourSelectedImage,yourSelectedImage_2, true);
                if (result == null)
                {
                    infoResult.setText("detect failed");
                }
                else
                {
                    String[] strs=result.split("-");
                    infoResult.setText(strs[0]);
                    timeResult.setText(strs[1]);

                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            try
            {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);
                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    yourSelectedImage = Bitmap.createScaledBitmap(rgba, 112, 112, false);
                    rgba.recycle();
                    imageView_1.setImageBitmap(bitmap);
                } else if (requestCode == SELECT_IMAGE_2) {
                    Bitmap bitmap = decodeUri(selectedImage);
                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    yourSelectedImage_2 = Bitmap.createScaledBitmap(rgba, 112, 112, false);
                    rgba.recycle();
                    imageView_2.setImageBitmap(bitmap);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
               || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

}
