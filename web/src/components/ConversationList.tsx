import { memo, useState } from "react";

export type ConversationInfo = {
  id: string;
  label: string;
  lastActiveAt: string;
};

type Props = {
  conversations: ConversationInfo[];
  activeId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, label: string) => void;
  disabled: boolean;
};

function formatRelativeTime(isoStr: string): string {
  try {
    const date = new Date(isoStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMin = Math.floor(diffMs / 60_000);
    if (diffMin < 1) return "刚刚";
    if (diffMin < 60) return `${diffMin} 分钟前`;
    const diffHrs = Math.floor(diffMin / 60);
    if (diffHrs < 24) return `${diffHrs} 小时前`;
    const diffDays = Math.floor(diffHrs / 24);
    if (diffDays < 30) return `${diffDays} 天前`;
    return date.toLocaleDateString();
  } catch {
    return "";
  }
}

export const ConversationList = memo(function ConversationList({
  conversations,
  activeId,
  onSelect,
  onCreate,
  onDelete,
  onRename,
  disabled,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editLabel, setEditLabel] = useState("");
  const [menuId, setMenuId] = useState<string | null>(null);

  const startRename = (id: string, currentLabel: string) => {
    setEditingId(id);
    setEditLabel(currentLabel);
    setMenuId(null);
  };

  const commitRename = () => {
    if (editingId && editLabel.trim()) {
      onRename(editingId, editLabel.trim());
    }
    setEditingId(null);
    setEditLabel("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      commitRename();
    } else if (e.key === "Escape") {
      setEditingId(null);
      setEditLabel("");
    }
  };

  return (
    <div className="conversation-list">
      <div className="conversation-list__header">
        <h2>会话</h2>
        <button
          type="button"
          className="conversation-list__new-btn"
          disabled={disabled}
          onClick={onCreate}
          title="新建会话"
        >
          +
        </button>
      </div>

      <ul className="conversation-list__items">
        {conversations.map((conv) => {
          const isActive = conv.id === activeId;
          const isEditing = editingId === conv.id;

          return (
            <li
              key={conv.id}
              className={`conversation-item${isActive ? " conversation-item--active" : ""}`}
            >
              {isEditing ? (
                <input
                  className="conversation-item__edit-input"
                  value={editLabel}
                  onChange={(e) => setEditLabel(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
              ) : (
                <button
                  type="button"
                  className="conversation-item__select-btn"
                  disabled={disabled}
                  onClick={() => onSelect(conv.id)}
                >
                  <span className="conversation-item__label">{conv.label}</span>
                  {conv.lastActiveAt && (
                    <span className="conversation-item__time">
                      {formatRelativeTime(conv.lastActiveAt)}
                    </span>
                  )}
                </button>
              )}

              <div className="conversation-item__actions">
                <button
                  type="button"
                  className="conversation-item__menu-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMenuId(menuId === conv.id ? null : conv.id);
                  }}
                  title="更多操作"
                >
                  ⋯
                </button>

                {menuId === conv.id && (
                  <div className="conversation-item__menu">
                    <button
                      type="button"
                      onClick={() => startRename(conv.id, conv.label)}
                    >
                      重命名
                    </button>
                    <button
                      type="button"
                      className="conversation-item__menu-danger"
                      onClick={() => {
                        setMenuId(null);
                        onDelete(conv.id);
                      }}
                    >
                      删除
                    </button>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      {conversations.length === 0 && (
        <p className="conversation-list__empty">暂无会话</p>
      )}
    </div>
  );
});
