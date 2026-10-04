const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const HOST = "127.0.0.1";
const PORT = 8765;
const URL = "http://" + HOST + ":" + PORT;

let backend = null;
let mainWindow = null;

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

function waitForBackend(timeoutMs = 60000) {
  const started = Date.now();

  return new Promise((resolve, reject) => {
    const retry = () => {
      if (Date.now() - started >= timeoutMs) {
        reject(new Error("Lucid backend did not start in time."));
        return;
      }
      setTimeout(probe, 350);
    };

    const probe = () => {
      const request = http.get(URL + "/health", (response) => {
        response.resume();
        if (response.statusCode && response.statusCode < 500) {
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

function startBackend() {
  const storage = path.join(app.getPath("userData"), "data");

  backend = spawn(
    backendPath(),
    ["--host", HOST, "--port", String(PORT)],
    {
      env: {
        ...process.env,
        LUCID_STORAGE_DIR: storage
      },
      windowsHide: true,
      stdio: "ignore"
    }
  );

  backend.on("exit", (code) => {
    if (!app.isQuitting && code !== 0 && mainWindow) {
      dialog.showErrorBox(
        "Lucid backend stopped",
        "The local Lucid service exited unexpectedly. Restart Lucid to try again."
      );
    }
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 900,
    minHeight: 650,
    backgroundColor: "#070707",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://") || url.startsWith("http://")) {
      shell.openExternal(url);
    }
    return { action: "deny" };
  });

  mainWindow.loadURL(URL);
}

app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.whenReady().then(async () => {
  try {
    startBackend();
    await waitForBackend();
    createWindow();
  } catch (error) {
    dialog.showErrorBox(
      "Lucid could not start",
      error instanceof Error ? error.message : String(error)
    );
    app.quit();
  }
});

app.on("before-quit", () => {
  app.isQuitting = true;
  if (backend && !backend.killed) {
    backend.kill();
  }
});

app.on("window-all-closed", () => {
  app.quit();
});
