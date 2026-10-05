const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");
const fs = require("fs");
const { autoUpdater } = require("electron-updater");

const HOST = "127.0.0.1";

let backend = null;
let mainWindow = null;
let backendUrl = null;
let backendLogHandle = null;
let startupWindow = null;
let backendStartError = null;

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
}

function backendPath() {
  const executable = process.platform === "win32"
    ? "lucid-backend.exe"
    : "lucid-backend";
  return path.join(process.resourcesPath, "backend", executable);
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, HOST, () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : null;
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        if (!port) {
          reject(new Error("Could not allocate a local Lucid port."));
          return;
        }
        resolve(port);
      });
    });
  });
}

function waitForBackend(url, timeoutMs = 60000) {
  const started = Date.now();

  return new Promise((resolve, reject) => {
    const retry = () => {
      if (backendStartError) {
        reject(
          new Error(
            "Lucid could not launch its local EEG engine: " +
            backendStartError.message
          )
        );
        return;
      }
      if (backend && backend.exitCode !== null) {
        reject(
          new Error(
            "Lucid's local EEG engine exited during startup. See the Lucid logs folder for details."
          )
        );
        return;
      }
      if (Date.now() - started >= timeoutMs) {
        reject(new Error("Lucid backend did not start in time."));
        return;
      }
      setTimeout(probe, 350);
    };

    const probe = () => {
      const request = http.get(url + "/health", (response) => {
        response.resume();
        if (response.statusCode === 200) {
          resolve();
          return;
        }
        retry();
      });

      request.on("error", retry);
      request.setTimeout(1500, () => {
        request.destroy();
        retry();
      });
    };

    probe();
  });
}

function appendDesktopLog(message) {
  try {
    const logs = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(logs, { recursive: true });
    fs.appendFileSync(
      path.join(logs, "lucid-desktop.log"),
      new Date().toISOString() + " " + message + "\n"
    );
  } catch {
    // Logging must never stop the app from starting.
  }
}

function createStartupWindow() {
  startupWindow = new BrowserWindow({
    width: 520,
    height: 300,
    resizable: false,
    maximizable: false,
    minimizable: false,
    show: false,
    backgroundColor: "#070707",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  const html = encodeURIComponent(`
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          html,body{margin:0;width:100%;height:100%;background:#070707;color:#f5f5f5;font-family:Arial,sans-serif}
          body{display:grid;place-items:center}
          main{width:78%}
          h1{font-size:34px;letter-spacing:.18em;margin:0 0 18px;font-weight:600}
          p{color:#9b9b9b;margin:0 0 18px;line-height:1.55}
          .bar{height:2px;background:#222;overflow:hidden}
          .bar:after{content:"";display:block;height:100%;width:38%;background:#ddd;animation:load 1.1s ease-in-out infinite alternate}
          @keyframes load{from{transform:translateX(-100%)}to{transform:translateX(260%)}}
        </style>
      </head>
      <body>
        <main>
          <h1>LUCID</h1>
          <p>Starting local EEG engine…</p>
          <div class="bar"></div>
        </main>
      </body>
    </html>
  `);

  startupWindow.loadURL("data:text/html;charset=utf-8," + html);
  startupWindow.once("ready-to-show", () => startupWindow.show());
}

function startBackend(port) {
  const storage = path.join(app.getPath("userData"), "data");
  const logs = path.join(app.getPath("userData"), "logs");
  fs.mkdirSync(logs, { recursive: true });
  const backendLog = path.join(logs, "lucid-backend.log");
  backendLogHandle = fs.openSync(backendLog, "a");

  const executable = backendPath();
  appendDesktopLog("backend path=" + executable);
  if (!fs.existsSync(executable)) {
    throw new Error(
      "Lucid backend executable is missing from the installation. Reinstall Lucid."
    );
  }

  backend = spawn(
    executable,
    ["--host", HOST, "--port", String(port)],
    {
      env: {
        ...process.env,
        LUCID_STORAGE_DIR: storage,
        LUCID_DIAGNOSTIC_LOG: path.join(logs, "lucid-backend-crash.log")
      },
      windowsHide: true,
      stdio: ["ignore", backendLogHandle, backendLogHandle]
    }
  );

  backend.on("error", (error) => {
    backendStartError = error;
    appendDesktopLog("backend spawn error: " + String(error));
  });

  backend.on("exit", (code, signal) => {
    appendDesktopLog(
      "backend exit code=" + String(code) + " signal=" + String(signal)
    );
    if (!app.isQuitting && code !== 0 && mainWindow) {
      dialog.showErrorBox(
        "Lucid backend stopped",
        "The local Lucid service exited unexpectedly. Restart Lucid to try again."
      );
    }
  });
}

function configureAutoUpdater() {
  if (!app.isPackaged) return;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  autoUpdater.on("error", (error) => {
    appendDesktopLog("update error: " + String(error));
  });
  autoUpdater.on("update-available", (info) => {
    appendDesktopLog("update available: " + String(info.version));
  });
  autoUpdater.on("update-downloaded", (info) => {
    appendDesktopLog("update downloaded: " + String(info.version));
  });

  setTimeout(() => {
    autoUpdater.checkForUpdatesAndNotify().catch(() => {});
  }, 5000);
}

function createWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 900,
    minHeight: 650,
    show: false,
    backgroundColor: "#070707",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  mainWindow.webContents.setWindowOpenHandler(({ url: externalUrl }) => {
    if (externalUrl.startsWith("https://")) {
      shell.openExternal(externalUrl);
    }
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, navigationUrl) => {
    if (!navigationUrl.startsWith(url)) {
      event.preventDefault();
      if (navigationUrl.startsWith("https://")) {
        shell.openExternal(navigationUrl);
      }
    }
  });

  mainWindow.webContents.on(
    "did-fail-load",
    (_event, errorCode, errorDescription) => {
      appendDesktopLog(
        "renderer load failure " +
        String(errorCode) +
        ": " +
        String(errorDescription)
      );
    }
  );

  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    appendDesktopLog(
      "renderer process gone: " +
      String(details.reason) +
      " exitCode=" +
      String(details.exitCode)
    );
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    if (startupWindow && !startupWindow.isDestroyed()) {
      startupWindow.close();
    }
    startupWindow = null;
  });

  mainWindow.loadURL(url);
}

app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

process.on("uncaughtException", (error) => {
  appendDesktopLog("uncaught exception: " + String(error?.stack || error));
  try {
    dialog.showErrorBox(
      "Lucid encountered an error",
      error instanceof Error ? error.message : String(error)
    );
  } catch {}
});

process.on("unhandledRejection", (error) => {
  appendDesktopLog("unhandled rejection: " + String(error));
});

app.whenReady().then(async () => {
  const selfTest = process.argv.includes("--self-test");
  if (!selfTest) {
    createStartupWindow();
  }
  try {
    const port = await findFreePort();
    backendUrl = "http://" + HOST + ":" + port;
    startBackend(port);
    await waitForBackend(backendUrl);

    if (selfTest) {
      appendDesktopLog(
        "Lucid packaged desktop self-test passed on local port " + String(port)
      );
      app.quit();
      return;
    }

    createWindow(backendUrl);
    appendDesktopLog("Lucid desktop started on local port " + String(port));
    configureAutoUpdater();
  } catch (error) {
    appendDesktopLog("startup failure: " + String(error?.stack || error));
    if (startupWindow && !startupWindow.isDestroyed()) {
      startupWindow.close();
      startupWindow = null;
    }
    const logs = path.join(app.getPath("userData"), "logs");
    dialog.showErrorBox(
      "Lucid could not start",
      (error instanceof Error ? error.message : String(error)) +
      "\n\nDiagnostics: " +
      logs
    );
    app.quit();
  }
});

app.on("before-quit", () => {
  app.isQuitting = true;
  if (startupWindow && !startupWindow.isDestroyed()) {
    startupWindow.destroy();
  }
  if (backend && !backend.killed) {
    backend.kill();
  }
  if (backendLogHandle !== null) {
    try {
      fs.closeSync(backendLogHandle);
    } catch {
      // Ignore shutdown logging errors.
    }
    backendLogHandle = null;
  }
});

app.on("window-all-closed", () => {
  app.quit();
});
