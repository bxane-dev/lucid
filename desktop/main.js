const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");

const HOST = "127.0.0.1";

let backend = null;
let mainWindow = null;
let backendUrl = null;

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

function startBackend(port) {
  const storage = path.join(app.getPath("userData"), "data");

  backend = spawn(
    backendPath(),
    ["--host", HOST, "--port", String(port)],
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

function createWindow(url) {
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

  mainWindow.webContents.setWindowOpenHandler(({ url: externalUrl }) => {
    if (
      (externalUrl.startsWith("https://") || externalUrl.startsWith("http://")) &&
      !externalUrl.startsWith(url)
    ) {
      shell.openExternal(externalUrl);
    }
    return { action: "deny" };
  });

  mainWindow.loadURL(url);
}

app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.whenReady().then(async () => {
  try {
    const port = await findFreePort();
    backendUrl = "http://" + HOST + ":" + port;
    startBackend(port);
    await waitForBackend(backendUrl);
    createWindow(backendUrl);
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
