# Push code directly to GPUHub (no GitHub pull on server)

## One-time setup on the server

SSH in, then in the repo that matches this remote path (`/root/autodl-tmp/workspace/RAG_Lab`):

```bash
cd /root/autodl-tmp/workspace/RAG_Lab
git config receive.denyCurrentBranch updateInstead
```

This allows `git push` into a non-bare repo and updates the checked-out files on `main` (Git 2.3+).

Ensure the directory is a normal clone (`git status` works). If it was copied without `.git`, clone from GitHub once on the server, then use that path in the remote URL below.

## One-time setup on your laptop (this repo)

Remote already added as `gpuhub`:

```text
ssh://root@connect.singapore-b.gpuhub.com:17754/root/autodl-tmp/workspace/RAG_Lab
```

If the path on the server differs, fix it with:

```bash
git remote set-url gpuhub ssh://root@connect.singapore-b.gpuhub.com:17754/PATH/TO/RAG_Lab
```

## Day-to-day

```bash
git push gpuhub main
# still push to GitHub when you want:
git push origin main
```

Use the same branch name on both sides (`main`).

## SSH without typing `-p 17754` every time

`~/.ssh/config`:

```sshconfig
Host gpuhub-sg
  HostName connect.singapore-b.gpuhub.com
  Port 17754
  User root
```

Then:

```bash
git remote set-url gpuhub ssh://gpuhub-sg/root/autodl-tmp/workspace/RAG_Lab
```

## Troubleshooting

- **Permission denied (publickey)**: add your laptop’s SSH public key to the server (`~/.ssh/authorized_keys` on GPUHub).
- **Could not read from remote**: wrong path or port; check `ssh -p 17754 root@connect.singapore-b.gpuhub.com` works.
- **refusing to update checked out branch**: run `git config receive.denyCurrentBranch updateInstead` on the server again.
