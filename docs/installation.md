# Install Colabinet and connect your Google account

This guide takes a new macOS installation from an empty terminal to a working Colabinet console. Colabinet
runs on your Mac; the inference engine and model run on a Google Colab runtime. You do not need CUDA,
PyTorch, or a local GPU on the Mac.

Installation, Google sign-in, and installation checks do not allocate a Colab runtime. Creating a session
later is a separate action that can consume your Google account's compute units (CU).

## 1. Prepare your Mac

You need:

| Requirement | Purpose |
| --- | --- |
| macOS | The supported Colabinet host; notifications use macOS facilities. |
| Git | Download and update this repository. |
| Python 3.12 or newer | Run Colabinet. The steps below install Python through uv. |
| uv | Manage Python, Colabinet dependencies, and the separate Colab CLI installation. |
| OpenSSH (`ssh` and `ssh-keygen`) | Establish the private tunnel to the Colab runtime. |
| A Google account with Colab access | Sign in to the Colab API and manage your own runtimes. |
| Paid Colab access and a positive CU balance, for real sessions | Colabinet uses SSH even for CPU sessions. |
| An internet connection | Install packages, authenticate, and reach Colab. |

Google restricts SSH on free managed runtimes without a positive CU balance. Real Colabinet sessions need
paid Colab access (such as Pro, Pro+, or Pay As You Go) and a positive compute-unit balance; exhausted
balances revert to free-tier restrictions. This applies to CPU sessions as well as GPUs. Installation,
sign-in, doctor checks, and the free console demo can be completed without allocating paid compute.
See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

In Terminal, check the tools supplied with macOS:

```bash
git --version
ssh -V
command -v ssh-keygen
```

The last command prints the path to `ssh-keygen`. If Git asks you to install Apple's command line developer
tools, complete that installation first (`xcode-select --install` also starts it).

Install uv using its [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new Terminal window so the installer's PATH change takes effect. Then run:

```bash
uv --version
uv python install 3.12
```

If you already use Homebrew, `brew install uv` is another supported way to install uv. Choose one method.

## 2. Install the official Colab CLI

Colabinet delegates Google account authentication and runtime management to Google's
[google-colab-cli](https://github.com/googlecolab/google-colab-cli). It requires a stable release **0.7.4 or newer**.
Install it as a separate uv tool, outside Colabinet's project environment:

```bash
uv tool install --python 3.12 --upgrade 'google-colab-cli>=0.7.4'
uv tool update-shell
```

Open another Terminal window, then verify the executable and version:

```bash
command -v colab
colab version
```

The first command prints the path to `colab`; the second prints `Version: ...`. The usual uv executable
directory is `~/.local/bin`, although uv allows a different tool directory.

If `colab` is not found, use `uv tool dir --bin` to locate uv's executable directory and follow the
`uv tool update-shell` instructions. Colabinet discovers `colab` on PATH, falls back to `~/.local/bin/colab`,
and also accepts `--colab-bin /absolute/path/to/colab`.

Installing Colabinet alone does **not** install `google-colab-cli`. Keep both installations up to date.

## 3. Download and install Colabinet

```bash
git clone https://github.com/cdownsimon/colabinet.git
cd colabinet
uv sync --frozen --python 3.12
uv run colabinet --version
```

If you already have a checkout, enter that directory and run the last two commands instead of cloning again.
`uv sync` creates a project-local `.venv` and installs the dependencies from `uv.lock`.

The rest of the source-checkout instructions use `uv run ...` from this repository directory.

## 4. Sign in with your Google account

You can complete sign-in through the web UI or Terminal. Both use the official Colab CLI; neither allocates
a runtime. The web UI saves separate account profiles. Terminal login uses the CLI's shared cached account.

### Sign in from the web UI

Start the local console:

```bash
uv run colabinet --stop-on-exit
```

Open **http://127.0.0.1:7860/settings**, then use the **Google accounts** card:

1. Wait for the initial account check. If you have already signed in, the card shows **Connected**, your
   account email and your compute-unit balance. **Check account** refreshes this check. On a fresh installation,
   a previously authenticated terminal CLI account appears as **Shared CLI account**.
2. Otherwise, select **Sign in with Google**. Wait for the **Open Google sign-in** link.
3. Open that link. In Google's new browser tab, choose your account, complete password/two-step verification,
   and review the requested permissions.
4. Copy the authorization code from Google's completion page. Return to the local console tab, paste it
   into **Authorization code**, and select **Connect account**.
5. Wait for **Connected** and confirm the displayed account email is the one you intended to use.

Google handles passwords and two-step verification. The code goes to the official CLI's waiting process;
Colabinet does not save it in the database or browser storage. **Cancel** stops the current sign-in attempt;
an unfinished attempt expires after 10 minutes. Invalid or expired codes show a retry message.

### Add accounts, switch, and log out

1. Select **Add account** in the Google accounts card. Each attempt uses a fresh profile and opens Google's
   account chooser, even when another account is already connected.
2. Complete the same Google link/code flow. The newly verified account becomes active. Existing accounts
   appear in the saved accounts table; signing in to the same email again keeps one account entry.
3. Select **Switch** beside another signed-in account. Colabinet verifies its identity and refreshes the
   active balance before changing the account used for runtime commands and SSH connections.
4. Select **Log out** to disconnect the active account. For an account added through the web UI, this removes
   its saved OAuth credential file. Other accounts remain signed in and can still be selected with **Switch**.
   A logged-out account requires the Google link/code flow again; **Check account** will not silently sign it back in.

Stop all managed sessions and wait for confirmed termination before adding a different account, switching,
or logging out. A pending creation, reserved name, or unconfirmed stop also blocks account changes. Sessions
retain their owning account in the database, and history remains available after switching. The daily CU cap
continues to cover managed sessions across all saved accounts; it does not reset when you switch.

**Sign in again** reauthenticates the active account and is available while its sessions still need supervision.
Choose the exact email shown in Settings. Colabinet rejects a different email and keeps the existing session
metadata. If a profile's credential file is missing after restart, the console stays available for this repair;
its account-dependent supervision resumes after successful authentication. Stop sessions in the Colab web UI
if you cannot restore account access.

Saved accounts and the selected account persist under the same `--data-dir`. Each private profile uses its own
OAuth credential, session store, CLI settings, and history beneath `<data-dir>/accounts/`. Account directories
are accessible only to your macOS user; metadata and credential files are private. Keep this directory out of
Git and shared folders. Passwords, authorization codes, and token contents are not returned in console pages
or stored in its SQLite database. Reauthentication transfers the verified credential into the existing profile
without changing its session store. Your actual home directory and SSH-agent configuration are preserved.

For a **Shared CLI account**, **Log out** disconnects this console and leaves
`~/.config/colab-cli/token.json` intact for other terminal uses. **Sign in again** for that shared account repairs
that shared credential file. To use isolated accounts going forward, use **Add account**. Local logout does not
sign you out of Google's browser pages or revoke Google's authorization grant.

The console can open before the SSH key is prepared. A missing-key alert is resolved in step 5; complete
the SSH and doctor steps before creating a session. After sign-in, stop this initial console with Control-C,
then continue below. Step 7 starts it again with your chosen SSH key.

For custom CLI/auth paths, pass the options in the optional section when starting the console. ADC mode
offers **Check account**; its sign-in is managed by the gcloud command described in that section.

### Sign in from Terminal

Alternatively, run:

```bash
uv run colabinet login
```

This delegates to `colab --auth=oauth2 usage` with the terminal's input and output attached. It requests
account usage information, which triggers sign-in when credentials have not been cached. It does not create
a VM and does not need an existing session.

Complete the following steps yourself:

1. Copy the Google authorization URL printed in the terminal and open it in your browser.
2. Select the Google account you want Colabinet to use. Complete any password or two-step verification prompts
   in Google's browser pages.
3. Review the requested permissions and approve if you want to grant the official CLI access.
4. Google displays an authorization code on its completion page. Copy that code.
5. Paste the code into the terminal at `Enter the authorization code:` and press Return.
6. Wait for the CLI to print your compute-unit balance, usage rate, and active assignment count.

The completion page may say **“Sign in to the gcloud CLI.”** The Colab CLI uses Google's bundled Cloud SDK
OAuth client and a Google-hosted completion page. This label is expected for the URL generated by the CLI.
The recommended OAuth2 flow needs no Google Cloud project, Cloud SDK installation, or custom OAuth client.
See Google's [authentication implementation](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/auth.py).

Terminal login caches OAuth credentials at `~/.config/colab-cli/token.json` and refreshes them on subsequent commands.
`colabinet login` always targets this shared terminal account, including when `--data-dir` has saved web accounts;
use **Sign in again** to repair a selected web account. Keep credentials and authorization codes private; paste the code
into your own local console's Authorization code field or your own terminal prompt.

If credentials are already cached, `login` prints usage without opening a new consent flow. To check the
cached account's identity, the CLI also provides:

```bash
colab --auth=oauth2 whoami
```

This is an upstream diagnostic command. Confirm the email is the Google account you intend to use.

**`colab auth` is a different operation:** it grants Google Cloud credentials to code inside an already
running VM. It is not the command for signing the local CLI in to your Google account.

## 5. Prepare an SSH key

Colabinet needs a supported SSH identity that can be used without an interactive passphrase prompt.
Ed25519 is recommended; the Colab SSH endpoint rejects RSA keys.

The default key path is `~/.ssh/id_ed25519`. If you already have an unencrypted Ed25519 key there, you can
use it and omit `--ssh-key` from the commands below. Otherwise, create a **dedicated** key for Colabinet:

```bash
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
ssh-keygen -t ed25519 -f "$HOME/.ssh/colabinet_ed25519" -N '' -C 'colabinet'
chmod 600 "$HOME/.ssh/colabinet_ed25519"
```

If that filename already exists, do not overwrite it: reuse the key if suitable or choose a new filename.
The empty `-N ''` creates a key without a passphrase so the background tunnel and the CLI's public-key
derivation can run unattended. Keep the private file local. The `.pub` file is the corresponding public key.

For a dedicated key, Colabinet passes its path to both OpenSSH and the Colab CLI; the CLI derives the public
key and sends it to the runtime's SSH endpoint. You do not need to manually install the public key in a Google dashboard.
See the [upstream SSH implementation](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/commands/ssh.py).

When using this dedicated key, repeat `--ssh-key "$HOME/.ssh/colabinet_ed25519"` for `doctor` and `serve`.
Infrastructure options apply to each invocation and are not saved by the Settings page.

Existing encrypted default keys also work when `~/.ssh/id_ed25519.pub` exists and the matching key is loaded
in your SSH agent. Unlock it in your terminal before running doctor or the console:

```bash
ssh-add "$HOME/.ssh/id_ed25519"
uv run colabinet doctor
uv run colabinet --stop-on-exit
```

For this default-key flow, the Colab CLI reads the public file and OpenSSH authenticates through the agent;
doctor checks that the agent contains the matching identity. Keep the agent available to the console.
Custom key paths use the CLI's private-key derivation, so the dedicated unencrypted key is the simplest setup.

## 6. Check the installation

From the repository directory:

```bash
uv run colabinet doctor --ssh-key "$HOME/.ssh/colabinet_ed25519"
```

`doctor` checks the CLI version, OpenSSH tools, private-key compatibility, and authenticated account API
access. It prints `OK` or `FAIL` with repair instructions and exits with status 0 on success, 1 on failure.
It does not start a console, create its database, or allocate a runtime. Its account check uses cached
credentials with terminal input disabled. With saved web accounts, it checks the selected account from the same
`--data-dir`; otherwise it checks the shared terminal account. Complete any needed consent in Settings first.

To check local dependencies before signing in, or without the account API request:

```bash
uv run colabinet doctor --offline --ssh-key "$HOME/.ssh/colabinet_ed25519"
```

`--offline` skips the account API check. A successful offline check does not verify Google sign-in, network
access, available compute units, or GPU entitlement. Even the full check verifies API access rather than
guaranteeing that a particular GPU can be allocated.

## 7. Start the console

```bash
uv run colabinet --ssh-key "$HOME/.ssh/colabinet_ed25519" --stop-on-exit
```

Open **http://127.0.0.1:7860** in your browser. Keep the terminal process running. On Settings, confirm the
CLI path, authentication method, and SSH key path, and choose your default idle limit and daily CU cap.

`--stop-on-exit` asks Colabinet to stop its managed sessions when you quit with Control-C. If you omit it,
Control-C leaves those runtimes running. Closing a browser tab does not quit Colabinet or stop a session.
The local inactivity guard needs the Colabinet process running and your Mac awake to enforce its limit.
Unmanaged runtime alerts do not schedule a stop. Stop those runtimes explicitly in Colab; do not rely on a
fixed idle-reclaim timeout.

If port 7860 is already occupied, choose another port:

```bash
uv run colabinet --port 7861 --ssh-key "$HOME/.ssh/colabinet_ed25519" --stop-on-exit
```

Then open `http://127.0.0.1:7861`. Terminal `ctl` commands must use the same `--port` and `--data-dir` as the console.

## 8. Create, use, and stop your first session

Google sign-in does not purchase a subscription or compute units. Confirm paid Colab access and a positive
balance in Overview before allocating any real session, including CPU. GPU availability depends on the account
and Google's current capacity; the doctor command does not perform a paid allocation test.

1. Open **Models** and choose a preset suitable for your intended GPU. Fit estimates are advisory.
2. Open **New**, select that preset and GPU, and choose a unique session name.
3. Set an idle stop and a per-session CU cap. The default idle stop is 30 minutes; the default daily cap is 10 CU.
4. Review the estimated rate and guard settings, then confirm **Create**. This is the step that allocates
   a runtime and can spend CU; setup and model downloads also consume runtime time.
5. Wait for the session to reach **ready**, then use **Playground** or copy its endpoint into your client.
6. When finished, select **Stop** and check that shutdown is confirmed. A “stop not confirmed” alert means
   the runtime may still be billing; follow the alert's instructions.

You can inspect and stop managed sessions from another terminal in the repository:

```bash
uv run colabinet ctl list
uv run colabinet ctl alerts
uv run colabinet ctl stop 1
```

Replace `1` with the session ID shown by `ctl list`. The stop command requests confirmation. Sessions created
outside Colabinet are not automatically adopted or stopped when its console exits.

## Optional: alternate paths and Google Cloud ADC

Use these options consistently for `login`, `doctor`, and `serve`:

| Option | Default | Use it when |
| --- | --- | --- |
| `--colab-bin PATH` | `colab` on PATH, then `~/.local/bin/colab` | You installed the CLI in another directory. |
| `--colab-auth oauth2` | `oauth2` | You want the direct Google account consent flow above. |
| `--colab-auth adc` | Opt-in | You already manage Google Application Default Credentials through gcloud. |
| `--ssh-key PATH` | `~/.ssh/id_ed25519` | You use a dedicated or differently named SSH key. |
| `--data-dir PATH` | `<checkout>/data` | You want the database and console logs in another directory. |

For example, a CLI installed in another directory works without editing source code:

```bash
uv run colabinet login --colab-bin /absolute/path/to/colab
uv run colabinet doctor --colab-bin /absolute/path/to/colab --ssh-key "$HOME/.ssh/colabinet_ed25519"
uv run colabinet --colab-bin /absolute/path/to/colab --ssh-key "$HOME/.ssh/colabinet_ed25519" --stop-on-exit
```

For ADC, first install the [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk) and authenticate
your Google user account with all of the scopes required by the Colab CLI:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
uv run colabinet login --colab-auth adc
uv run colabinet doctor --colab-auth adc --ssh-key "$HOME/.ssh/colabinet_ed25519"
uv run colabinet --colab-auth adc --ssh-key "$HOME/.ssh/colabinet_ed25519" --stop-on-exit
```

`gcloud auth login` alone does not configure ADC. In ADC mode, `colabinet login` checks the existing ADC
credentials; the interactive sign-in is handled by the preceding gcloud command. Colabinet passes the same
authentication method to runtime commands and its SSH bridge. See Google's
[authentication documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md).
Saved account switching and web logout are available in OAuth2 mode. ADC credentials are managed by gcloud;
use `gcloud auth application-default revoke` to remove those credentials. Use a separate `--data-dir` for ADC
and keep the authentication method consistent while managed sessions exist.

## Optional: install Colabinet as a standalone command

After completing setup in the checkout, you can install its packaged CLI and console:

```bash
uv tool install --python 3.12 .
uv tool update-shell
```

Open a new Terminal window and use `colabinet login`, `colabinet doctor`, and `colabinet ...` without
`uv run` or a checkout working directory. The packaged installation includes default presets and console
assets. Its state directory defaults to `~/.local/share/colabinet`; the source checkout uses `<checkout>/data`.
Use `--data-dir` explicitly if you want them to share state, and run only one console for that directory.

## Update and troubleshoot

If a session fails to create, open **Overview → View error** on its failed row, or **Sessions → History →
View error**. The session page shows the failure reason and the CLI exit code when available. Choose
**View logs** for the creation output or **Download log** to save it alongside the lifecycle and engine logs.
Creation output is saved when the CLI exits, survives console restarts, and retains up to 65,536 characters
with credentials redacted. Older failures display the short message retained in their history; output that
was previously discarded cannot be recovered.

Update the source installation from its checkout, then rerun the doctor with your usual key/auth options:

```bash
git pull --ff-only
uv sync --frozen
uv tool upgrade google-colab-cli
uv run colabinet doctor --ssh-key "$HOME/.ssh/colabinet_ed25519"
```

If you installed Colabinet as a standalone tool, refresh it from the updated checkout with
`uv tool install --reinstall .`.

| Symptom | What to check |
| --- | --- |
| `uv: command not found` | Open a new terminal after installing uv; follow the installer's PATH instructions. |
| `colab: command not found` or CLI missing | Install `google-colab-cli`, run `uv tool update-shell`, and reopen Terminal; use `uv tool dir --bin` or `--colab-bin` for custom locations. |
| Doctor rejects the CLI version | Install a stable release >= 0.7.4 with `uv tool install --upgrade 'google-colab-cli>=0.7.4'`. |
| No Google consent prompt during terminal login | A shared cached account is already signed in; use `colab --auth=oauth2 whoami`. Settings → Google accounts → Add account always opens a fresh consent flow. |
| Account changes are blocked | Stop every managed session and wait for confirmation, including pending creation or unconfirmed termination. Use Sign in again if the same account needs credential repair. |
| Saved account credentials are missing or expired | Use Sign in again for the active account, or Sign in with Google beside a logged-out saved account. Choose its displayed email. |
| Account profiles require the official Python CLI | Install using `uv tool install --upgrade 'google-colab-cli>=0.7.4'` and select its generated Python launcher with `--colab-bin`; shell wrappers and unrelated binaries are not supported for private profiles. |
| Web sign-in is cancelled, expired, or rejects the code | Select Sign in with Google again and use the new link/code. Keep the console running while completing Google consent. |
| Doctor says account API access failed | In Settings → Google accounts, select the intended saved account or use Sign in again with the same `--data-dir`. For initial shared CLI setup, use terminal `login`; check the network. For ADC, repeat the gcloud command with the complete scope list. |
| HTTP 401/403 | Check the selected account, auth method, granted scopes, and any account administrator restrictions. `colab auth` is not a local login repair. |
| OAuth `redirect_uri_mismatch` or missing client configuration | A custom `~/.colab-cli-oauth-config.json` can override Google's bundled client. Check whether you intentionally installed one and consult the upstream authentication documentation. |
| SSH key missing, encrypted, or RSA | Use the dedicated unencrypted Ed25519 key from step 5 and pass its path to both doctor and serve. |
| SSH HTTP 429 / already-active connection | Disconnect another SSH connection to the same runtime, then let Colabinet reconnect. |
| GPU request rejected / no available quota | Check the Google account's current Colab entitlement, balance, and accelerator availability. |
| Port already in use / another console running | Use a different `--port`, or stop the other console using the same data directory. |
| Session still running after quitting | Start with `--stop-on-exit`, or explicitly stop that session and confirm it disappears. |

Use **Settings → Google accounts → Add account / Switch / Log out** to manage console accounts.
Do not delete CLI session metadata or Colabinet's database as a login repair.

For a console preview before you allocate a real runtime, follow the [free demo instructions](../README.md#try-it-without-colab).
