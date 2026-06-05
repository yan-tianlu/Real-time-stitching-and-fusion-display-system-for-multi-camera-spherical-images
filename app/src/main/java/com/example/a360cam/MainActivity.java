package com.example.a360cam;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "360Cam_MJPEG";
    // 2. 更新视频流地址
    private static final String STREAM_URL = "http://172.20.10.5:5001/video";
    
    private final float sensitivity = 10.0f;
    private float lastYaw = 0f;
    private boolean isSensorInitialized = false;
    private float currentScrollX = 0f;
    private float totalRotatedDegrees = 0f; // 记录总旋转度数
    private boolean isFirstFrame = true;

    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;
    private TextView tvInfo;

    private SensorManager sensorManager;
    private Sensor rotationSensor;

    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.SECONDS)
            .build();
    private Call currentCall;
    private Thread mjpegThread;
    private volatile boolean isRunning = false;
    private volatile Bitmap currentBitmap;
    private final Paint paint = new Paint();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // 优化：开启抗锯齿和位图过滤，改善画面拉伸后的模糊感
        paint.setAntiAlias(true);
        paint.setFilterBitmap(true);
        paint.setDither(true);

        initViews();
        initSensors();
    }

    private void initViews() {
        surfaceView = findViewById(R.id.surfaceView);
        tvInfo = findViewById(R.id.tvInfo);
        Button btnReset = findViewById(R.id.btnReset);

        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                Log.d(TAG, "surfaceCreated - starting MJPEG stream");
                isRunning = true;
                startMjpegStream();
            }

            @Override
            public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {}

            @Override
            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
                isRunning = false;
                if (currentCall != null) {
                    currentCall.cancel();
                }
            }
        });

        btnReset.setOnClickListener(v -> {
            isFirstFrame = true;
            isSensorInitialized = false;
            totalRotatedDegrees = 0f; // 重置旋转角度显示
            updateStatus("视角已重置并归中");
        });
    }

    private void initSensors() {
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        }
        if (rotationSensor == null) {
            updateStatus("错误：不支持旋转矢量传感器");
        }
    }

    private void startMjpegStream() {
        if (isRunning && mjpegThread != null && mjpegThread.isAlive()) {
            Log.d(TAG, "MJPEG thread is already running.");
            return;
        }
        isRunning = true;
        mjpegThread = new Thread(() -> {
            Log.d(TAG, "Attempting to start MJPEG stream from: " + STREAM_URL);
            while (isRunning) {
                try {
                    Request request = new Request.Builder().url(STREAM_URL).build();
                    Log.d(TAG, "Creating new call to: " + STREAM_URL);
                    currentCall = client.newCall(request);
                    try (Response response = currentCall.execute()) {
                        Log.d(TAG, "Response received. Code: " + response.code() + ", Success: " + response.isSuccessful());
                        if (!response.isSuccessful() || response.body() == null) {
                            updateStatus("连接失败: " + response.code());
                            Thread.sleep(2000);
                            continue;
                        }

                        updateStatus("视频流已连接");
                        Log.d(TAG, "Stream connected, starting to read body...");
                        try (InputStream is = new BufferedInputStream(response.body().byteStream())) {
                            ByteArrayOutputStream frameBuffer = new ByteArrayOutputStream(1024 * 300);
                            int prevByte = -1;
                            int curByte;

                            while (isRunning && (curByte = is.read()) != -1) {
                                frameBuffer.write(curByte);
                                // 寻找 JPEG 结束标志 0xFF 0xD9
                                if (prevByte == 0xFF && curByte == 0xD9) {
                                    byte[] rawBytes = frameBuffer.toByteArray();
                                    
                                    // 3. 优化解码保护：寻找帧头，处理完后 reset
                                    int start = -1;
                                    for (int i = 0; i < rawBytes.length - 1; i++) {
                                        if ((rawBytes[i] & 0xFF) == 0xFF && (rawBytes[i + 1] & 0xFF) == 0xD8) {
                                            start = i;
                                            break;
                                        }
                                    }

                                    if (start != -1) {
                                        Bitmap bitmap = BitmapFactory.decodeByteArray(rawBytes, start, rawBytes.length - start);
                                        // 确认返回 null 时不触发绘制
                                        if (bitmap != null) {
                                            processNewFrame(bitmap);
                                        }
                                    }
                                    frameBuffer.reset(); // 确保处理完一帧后重置
                                }
                                prevByte = curByte;
                                if (frameBuffer.size() > 1024 * 1024) frameBuffer.reset();
                            }
                        }
                    }
                } catch (java.net.ConnectException e) {
                    if (isRunning) {
                        Log.e(TAG, "Connection failed: " + e.getMessage());
                        updateStatus("连接失败: 无法连接服务器");
                        try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
                    }
                } catch (Exception e) {
                    if (isRunning) {
                        Log.e(TAG, "MJPEG Thread error", e);
                        updateStatus("连接断开，正在重连...");
                        try { Thread.sleep(2000); } catch (InterruptedException ignored) {}
                    }
                }
            }
            Log.d(TAG, "MJPEG thread exiting");
        }, "360Cam_MJPEG");
        mjpegThread.start();
    }

    private void processNewFrame(Bitmap bmp) {
        int W = bmp.getWidth();
        int H = bmp.getHeight();
        int screenW = surfaceView.getWidth();
        int screenH = surfaceView.getHeight();

        if (screenW <= 0 || screenH <= 0) return;

        if (isFirstFrame) {
            float windowWidthInBitmap = screenW * ((float) H / screenH);
            currentScrollX = (W / 2.0f) - (windowWidthInBitmap / 2.0f);
            isFirstFrame = false;
        }

        Bitmap oldBitmap = currentBitmap;
        currentBitmap = bmp;
        if (oldBitmap != null && oldBitmap != bmp && !oldBitmap.isRecycled()) {
            oldBitmap.recycle();
        }
        
        drawToCanvas();
    }

    private synchronized void drawToCanvas() {
        Bitmap bitmap = currentBitmap;
        if (bitmap == null || !surfaceHolder.getSurface().isValid()) return;

        Canvas canvas = surfaceHolder.lockCanvas();
        if (canvas != null) {
            try {
                canvas.drawColor(Color.BLACK);
                
                int screenW = canvas.getWidth();
                int screenH = canvas.getHeight();
                int W = bitmap.getWidth();
                int H = bitmap.getHeight();

                float windowWidthInBitmap = screenW * ((float) H / screenH);
                
                int left = (int) currentScrollX;
                int right = left + (int) windowWidthInBitmap;

                // 强制防越界保护
                if (left < 0) {
                    left = 0;
                    right = (int) windowWidthInBitmap;
                }
                if (right > W) {
                    right = W;
                    left = Math.max(0, right - (int) windowWidthInBitmap);
                }
                
                currentScrollX = left;

                Rect srcRect = new Rect(left, 0, right, H);
                Rect dstRect = new Rect(0, 0, screenW, screenH);

                canvas.drawBitmap(bitmap, srcRect, dstRect, paint);
                
            } catch (Exception e) {
                Log.e(TAG, "Draw error", e);
            } finally {
                surfaceHolder.unlockCanvasAndPost(canvas);
            }
        }
    }

    private void updateStatus(final String msg) {
        runOnUiThread(() -> tvInfo.setText(msg));
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
            float[] rotationMatrix = new float[9];
            SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values);
            float[] orientation = new float[3];
            SensorManager.getOrientation(rotationMatrix, orientation);

            float currentYaw = (float) Math.toDegrees(orientation[0]);
            // Log.d(TAG, "onSensorChanged: yaw=" + currentYaw); // 可选，如果日志太多可以注释掉

            if (!isSensorInitialized) {
                lastYaw = currentYaw;
                isSensorInitialized = true;
                return;
            }

            float deltaYaw = currentYaw - lastYaw;
            if (deltaYaw > 180) deltaYaw -= 360;
            else if (deltaYaw < -180) deltaYaw += 360;

            // 1. 优化死区逻辑：增加阈值到 0.5 度，抑制明显的静态漂移
            if (Math.abs(deltaYaw) < 0.5) {
                deltaYaw = 0;
            } else {
                // 2. 平滑处理：对大幅度转动进行简单线性平滑，减少突发跳跃感
                deltaYaw = deltaYaw * 0.8f; 
            }

            // 3. 更新总旋转度数（仅在超过死区时）
            if (deltaYaw != 0) {
                totalRotatedDegrees += deltaYaw;
                currentScrollX += deltaYaw * sensitivity;
            }
            
            lastYaw = currentYaw;
            
            // 4. 降低 UI 刷新频率感，仅在变化时更新文本
            if (deltaYaw != 0) {
                tvInfo.setText(String.format(Locale.getDefault(), 
                        "当前视角: %.1f° | 状态: 移动中", totalRotatedDegrees));
            } else {
                tvInfo.setText(String.format(Locale.getDefault(), 
                        "当前视角: %.1f° | 状态: 静止锁定", totalRotatedDegrees));
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    protected void onResume() {
        super.onResume();
        if (rotationSensor != null) {
            sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_UI);
        }
        // 如果 Surface 已经就绪但线程没在跑，则启动（处理 onPause 后恢复的情况）
        if (surfaceHolder.getSurface().isValid()) {
            startMjpegStream();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "onPause called - stopping stream");
        isRunning = false;
        if (currentCall != null) {
            currentCall.cancel();
        }
        sensorManager.unregisterListener(this);
    }
}
